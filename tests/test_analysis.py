import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from analyse import summarise
from waterbirds_core import (METHODS, early_scores, group_metrics, select_examples,
                             write_csv, write_json)


def fixture(root):
    c = json.loads((Path(__file__).resolve().parents[1]/"configs/smoke.json").read_text())
    identity = dict(config=c,seed=42,dataset=dict(kind="synthetic",n_train=4,n_val=4,n_test=4),
                    versions={},source_hashes={},device="cpu",weights_sha256=None,python="fixture")
    seed = root/"seed_42"
    seed.mkdir()
    write_json(seed/"metadata.json",dict(identity=identity))
    signature=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    rows=[dict(seed=42,epoch=e,sample_id=i,per_example_loss=float(i+1),correct=i%2)
          for e in (1,2) for i in range(4)]
    scores=early_scores(rows,range(4),1)
    write_csv(seed/"training_metadata.csv",[dict(sample_id=i,label=i%2,background=i//2) for i in range(4)])
    for method in METHODS:
        folder=seed/method
        folder.mkdir()
        (folder/"COMPLETE").write_text("done")
        selected=select_examples(scores,method,c["fraction"],42)
        write_json(folder/"selection.json",dict(selected_ids=selected,subgroup_metadata_used=False))
        write_csv(folder/"epoch_metrics.csv",[dict(epoch=1,validation_accuracy=.5),dict(epoch=2,validation_accuracy=.75)])
        result=dict(config=c,seed=42,method=method,signature=signature,selected_epoch=2,
                    initial_model_sha256="same")
        for split,start in (("validation",100),("test",200)):
            predictions=[dict(sample_id=start+i,label=i%2,background=i//2,
                              prediction=0 if split=="validation" and i==3 else i%2) for i in range(4)]
            write_csv(folder/f"{split}_predictions.csv",predictions)
            result[split]=group_metrics(predictions)
        write_json(folder/"result.json",result)
        if method=="erm":
            write_csv(folder/"training_dynamics.csv",rows)
    (seed/"COMPLETE").write_text("done")
    return c


class AnalysisTests(unittest.TestCase):
    def test_complete_comparison_and_smoke_separation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            c=fixture(root)
            rows,summary,paired,detection=summarise(root,c,allow_synthetic=True)
            self.assertEqual(len(rows),4)
            self.assertEqual(len(summary),4)
            self.assertTrue(all(r["worst_group_delta_pp"]==0 for r in paired))
            self.assertEqual(len(detection),1)
            with self.assertRaisesRegex(ValueError,"Synthetic"):
                summarise(root,c)

    def test_missing_method_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            c=fixture(root)
            (root/"seed_42/jtt/COMPLETE").unlink()
            with self.assertRaisesRegex(ValueError,"Incomplete method"):
                summarise(root,c,True)

    def test_altered_selection_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            c=fixture(root)
            write_json(root/"seed_42/early_loss/selection.json",dict(selected_ids=[999],subgroup_metadata_used=False))
            with self.assertRaisesRegex(ValueError,"Selection"):
                summarise(root,c,True)

    def test_wrong_metrics_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            c=fixture(root)
            path=root/"seed_42/erm/result.json"
            result=json.loads(path.read_text())
            result["test"]["worst_group_accuracy"]=.7
            write_json(path,result)
            with self.assertRaisesRegex(ValueError,"metrics"):
                summarise(root,c,True)


if __name__ == "__main__":
    unittest.main()
