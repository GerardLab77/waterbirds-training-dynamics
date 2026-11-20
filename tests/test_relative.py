import copy
import json
import tempfile
import unittest
from pathlib import Path

from analyse_relative import summarise_relative
from relative_rules import (RELATIVE_METHODS, class_counts, digest_json, load_reference,
                            make_plan, validate_relative_config, validate_selections,
                            weighted_class_totals)
from waterbirds_core import group_metrics, write_csv, write_json


def histories():
    return [dict(seed=42,epoch=e,sample_id=i,label=i//3,
                 per_example_loss=float((i%3+1)*(1 if i<3 else 100)),correct=i%2)
            for e in (1,2) for i in range(6)]


def fixture(root):
    project=Path(__file__).resolve().parents[1]
    c=json.loads((project/"configs/relative-smoke.json").read_text())
    baseline=json.loads((project/"configs/smoke.json").read_text())
    reference=root/"reference"
    seed=reference/"seed_42"
    erm=seed/"erm"
    erm.mkdir(parents=True)
    (erm/"COMPLETE").write_text("done")
    dataset=dict(kind="synthetic",n_train=6,n_val=4,n_test=4)
    env=dict(versions={},python="fixture",device="cpu",weights_sha256=None)
    initial="a"*64
    old_identity=dict(config=baseline,seed=42,dataset=dataset,source_hashes={},**env)
    write_json(seed/"metadata.json",dict(identity=old_identity))
    history=[dict(epoch=1,validation_accuracy=.5),dict(epoch=2,validation_accuracy=.75)]
    write_csv(erm/"epoch_metrics.csv",history)
    write_csv(erm/"training_dynamics.csv",histories())
    result=dict(seed=42,method="erm",config=baseline,selected_epoch=2,
                initial_model_sha256=initial,signature=digest_json(old_identity))
    predictions={}
    for split,start in (("validation",100),("test",200)):
        predictions[split]=[dict(sample_id=start+i,label=i%2,background=i//2,
                                 prediction=0 if i==3 else i%2) for i in range(4)]
        write_csv(erm/f"{split}_predictions.csv",predictions[split])
        result[split]=group_metrics(predictions[split])
    write_json(erm/"result.json",result)
    plan=load_reference(reference,42,c)["plan"]
    run=root/"relative"
    new_seed=run/"seed_42"
    new_seed.mkdir(parents=True)
    identity=dict(experiment="class-relative-retraining-v1",config=c,seed=42,
        dataset=dataset,source_hashes={},selection_plan_sha256=digest_json(plan),
        reference_initial_model_sha256=initial,**env)
    write_json(new_seed/"metadata.json",dict(identity=identity))
    write_json(new_seed/"selection_plan.json",plan)
    for method in RELATIVE_METHODS:
        folder=new_seed/method
        folder.mkdir()
        (folder/"COMPLETE").write_text("done")
        write_csv(folder/"epoch_metrics.csv",history)
        write_json(folder/"selection.json",dict(method=method,selected_ids=plan["selected"][method],
            early_epoch=c["early_epoch"],fraction=c["fraction"],upweight=c["upweight"],
            subgroup_metadata_used=False))
        record=dict(seed=42,method=method,config=c,selected_epoch=2,
            selected_count=plan["selected_count"],initial_model_sha256=initial,
            signature=digest_json(identity))
        for split in ("validation","test"):
            values=copy.deepcopy(predictions[split])
            if split=="test" and method=="class_relative":
                values[-1]["prediction"]=values[-1]["label"]
            write_csv(folder/f"{split}_predictions.csv",values)
            record[split]=group_metrics(values)
        write_json(folder/"result.json",record)
    (new_seed/"COMPLETE").write_text("done")
    return c,reference,run


class RelativeRuleTests(unittest.TestCase):
    def test_exact_class_counts_and_weighted_mass_match(self):
        plan=make_plan(histories(),range(6),2,.5,10042,10)
        scores=[dict(sample_id=i,label=i//3) for i in range(6)]
        first,second=(plan["selected"][m] for m in RELATIVE_METHODS)
        self.assertEqual(class_counts(scores,first),class_counts(scores,second))
        self.assertEqual(weighted_class_totals(scores,first,10),weighted_class_totals(scores,second,10))
        self.assertEqual(len(first),3)
        self.assertEqual(len(first),len(second))

    def test_random_selection_is_seeded_and_order_independent(self):
        a=make_plan(histories(),range(6),2,.5,10042,10)
        b=make_plan(list(reversed(histories())),list(reversed(range(6))),2,.5,10042,10)
        self.assertEqual(a,b)

    def test_background_fields_cannot_change_selection(self):
        rows=histories()
        expected=make_plan(rows,range(6),2,.5,10042,10)
        for row in rows:
            row.update(background="must not read",group_id=object())
        self.assertEqual(make_plan(rows,range(6),2,.5,10042,10),expected)

    def test_ties_use_sample_id_and_zero_class_quota_works(self):
        rows=histories()
        for row in rows:
            row["per_example_loss"]=1
        plan=make_plan(rows,range(6),2,1/6,10042,10)
        self.assertEqual(plan["selected"]["class_relative"],[0])
        self.assertEqual(plan["class_counts"],{"0":1,"1":0})
        self.assertEqual(len(plan["selected"]["class_matched_random"]),1)

    def test_duplicate_or_unknown_selected_ids_fail(self):
        scores=[dict(sample_id=i,label=i//3) for i in range(6)]
        for selected in ([0,0],[999]):
            with self.assertRaises(ValueError):
                class_counts(scores,selected)

    def test_wrong_control_cannot_pass_as_matched(self):
        plan=make_plan(histories(),range(6),2,.5,10042,10)
        actual=copy.deepcopy(plan["selected"])
        actual["class_matched_random"]=[999]
        with self.assertRaisesRegex(ValueError,"Selection differs"):
            validate_selections(plan,actual)

    def test_main_budget_must_match_reference(self):
        c=json.loads((Path(__file__).resolve().parents[1]/"configs/relative.json").read_text())
        validate_relative_config(c)
        c["epochs"]=10
        with self.assertRaisesRegex(ValueError,"epoch budget"):
            validate_relative_config(c)

    def test_reference_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            c,reference,_=fixture(Path(tmp))
            c["learning_rate"]*=2
            with self.assertRaisesRegex(ValueError,"Reference setting"):
                load_reference(reference,42,c)


class RelativeAnalysisTests(unittest.TestCase):
    def test_primary_comparison_is_scored_minus_matched_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            c,reference,run=fixture(Path(tmp))
            rows,summary,paired=summarise_relative(run,reference,c,True)
            self.assertEqual(len(rows),3)
            self.assertEqual(len(summary),3)
            self.assertEqual(paired[0]["worst_group_delta_pp"],100)
            self.assertEqual(paired[0]["overall_delta_pp"],25)
            with self.assertRaisesRegex(ValueError,"Synthetic"):
                summarise_relative(run,reference,c)

    def test_missing_method_is_not_silently_averaged(self):
        with tempfile.TemporaryDirectory() as tmp:
            c,reference,run=fixture(Path(tmp))
            (run/"seed_42/class_matched_random/COMPLETE").unlink()
            with self.assertRaisesRegex(ValueError,"Incomplete method"):
                summarise_relative(run,reference,c,True)

    def test_modified_selection_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            c,reference,run=fixture(Path(tmp))
            path=run/"seed_42/class_matched_random/selection.json"
            value=json.loads(path.read_text())
            value["selected_ids"]=[999]
            write_json(path,value)
            with self.assertRaisesRegex(ValueError,"Selection differs"):
                summarise_relative(run,reference,c,True)

    def test_different_initial_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            c,reference,run=fixture(Path(tmp))
            path=run/"seed_42/class_relative/result.json"
            value=json.loads(path.read_text())
            value["initial_model_sha256"]="wrong"
            write_json(path,value)
            with self.assertRaisesRegex(ValueError,"initial model"):
                summarise_relative(run,reference,c,True)


if __name__=="__main__":
    unittest.main()
