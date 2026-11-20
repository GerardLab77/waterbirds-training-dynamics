import copy
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from waterbirds_core import (best_epoch, detection_metrics, early_scores, group_metrics,
                             sample_weights, select_examples, validate_config)
from prepare_waterbirds import safe_extract


def dynamics():
    return [dict(seed=42, epoch=e, sample_id=i, per_example_loss=float(i+1), correct=int(i%2 == 0))
            for e in (1,2) for i in (10,11,12,13)]


class RuleTests(unittest.TestCase):
    def test_stable_ids_and_early_window(self):
        rows = dynamics() + [dict(seed=42, epoch=3, sample_id=10, per_example_loss=999, correct=0)]
        scores = early_scores(list(reversed(rows)), [13,12,11,10], 2)
        self.assertEqual([r["sample_id"] for r in scores], [10,11,12,13])
        self.assertEqual(scores[0]["mean_loss"], 11)

    def test_metadata_cannot_change_scores(self):
        rows = dynamics()
        expected = early_scores(rows, [10,11,12,13], 2)
        for row in rows:
            row.update(background=999, group_id="do not read", test_accuracy=1)
        self.assertEqual(early_scores(rows, [10,11,12,13], 2), expected)

    def test_rejects_missing_duplicate_and_mixed_rows(self):
        for rows in (dynamics()[:-1], dynamics()+[dynamics()[0]],
                     [dict(r,seed=43) if i==0 else r for i,r in enumerate(dynamics())]):
            with self.assertRaises(ValueError):
                early_scores(rows, [10,11,12,13], 2)

    def test_loss_ranking_and_jtt_use_training_only(self):
        scores = early_scores(dynamics(), [10,11,12,13], 2)
        self.assertEqual(select_examples(scores,"early_loss",.5,42), [13,12])
        self.assertEqual(select_examples(scores,"jtt",.5,42), [11,13])

    def test_ties_are_resolved_by_id(self):
        scores = [dict(sample_id=i,mean_loss=1,early_error=0) for i in (13,10,12,11)]
        self.assertEqual(select_examples(scores,"early_loss",.5,42), [10,11])
        self.assertEqual(select_examples(scores,"jtt",.5,42), [])

    def test_random_control_matches_count_and_seed(self):
        scores = early_scores(dynamics(), [10,11,12,13], 2)
        selected = select_examples(scores,"random",.5,42)
        self.assertEqual(selected, select_examples(list(reversed(scores)),"random",.5,42))
        self.assertEqual(len(selected),len(select_examples(scores,"early_loss",.5,42)))

    def test_weights_follow_id_not_position(self):
        self.assertEqual(sample_weights([13,10,12,11],[10,11],10), [1,10,1,10])
        with self.assertRaises(ValueError):
            sample_weights([1,2],[3],10)

    def test_checkpoint_selection_ignores_subgroups_and_test(self):
        rows = [dict(epoch=1,validation_accuracy=.8,test_accuracy=1,worst_group_accuracy=1),
                dict(epoch=2,validation_accuracy=.9,test_accuracy=0,worst_group_accuracy=0),
                dict(epoch=3,validation_accuracy=.9,test_accuracy=1,worst_group_accuracy=1)]
        self.assertEqual(best_epoch(rows)["epoch"],2)

    def test_group_metrics_require_all_groups(self):
        rows = [dict(sample_id=i,label=i%2,background=i//2,prediction=(i%2 if i!=3 else 0)) for i in range(4)]
        self.assertEqual(group_metrics(rows)["overall_accuracy"], .75)
        self.assertEqual(group_metrics(rows)["worst_group_accuracy"],0)
        for invalid in (rows[:-1],rows+[rows[0]]):
            with self.assertRaises(ValueError):
                group_metrics(invalid)

    def test_tied_detection_scores_have_chance_auc(self):
        scores = [dict(sample_id=i,mean_loss=1) for i in range(4)]
        metadata = [dict(sample_id=i,label=i%2,background=i//2) for i in range(4)]
        result = detection_metrics(scores,metadata)
        self.assertEqual(result["roc_auc"],.5)
        self.assertEqual(result["average_precision"],.5)

    def test_perfect_detection(self):
        metadata = [dict(sample_id=i,label=i%2,background=i//2) for i in range(4)]
        scores = [dict(sample_id=i,mean_loss=int(r["label"]!=r["background"])) for i,r in enumerate(metadata)]
        result = detection_metrics(scores,metadata)
        self.assertEqual(result["roc_auc"],1)
        self.assertEqual(result["average_precision"],1)

    def test_invalid_config(self):
        root = Path(__file__).resolve().parents[1]
        c = json.loads((root/"configs/comparison.json").read_text())
        validate_config(c)
        for key,value in (("early_epoch",31),("fraction",0),("upweight",float('nan')),
                          ("seeds",[42,42]),("pretrained",0)):
            invalid=copy.deepcopy(c)
            invalid[key]=value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config(invalid)

    def test_archive_rejects_traversal_and_links(self):
        for name, kind in (("../escape",tarfile.REGTYPE),("link",tarfile.SYMTYPE)):
            stream=io.BytesIO()
            with tarfile.open(fileobj=stream,mode="w") as archive:
                entry=tarfile.TarInfo(name)
                entry.type=kind
                archive.addfile(entry)
            stream.seek(0)
            with tempfile.TemporaryDirectory() as tmp, tarfile.open(fileobj=stream) as archive:
                with self.assertRaises(ValueError):
                    safe_extract(archive,Path(tmp))


if __name__ == "__main__":
    unittest.main()
