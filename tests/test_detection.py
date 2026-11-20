import unittest

from detect import percentile_ranks, score_examples, evaluate


def histories():
    return [dict(seed=42,epoch=e,sample_id=i,label=i//3,
                 per_example_loss=(i%3+1)*(1 if i<3 else 100),correct=0)
            for e in (1,2) for i in range(6)]


class DetectorTests(unittest.TestCase):
    def test_relative_scores_remove_class_loss_scale(self):
        scores=score_examples(histories(),range(6),2)
        self.assertEqual([r['class_relative_loss'] for r in scores],[0,.5,1,0,.5,1])

    def test_subgroup_metadata_never_changes_scores(self):
        rows=histories()
        expected=score_examples(rows,range(6),2)
        for row in rows:
            row.update(background='unreadable',group_id=999)
        self.assertEqual(score_examples(rows,range(6),2),expected)

    def test_ties_and_singletons_remain_neutral(self):
        self.assertEqual(percentile_ranks({0:7,1:7,2:7}),{0:.5,1:.5,2:.5})
        self.assertEqual(percentile_ranks({0:99}),{0:.5})

    def test_input_order_does_not_change_ranks(self):
        rows=histories()
        self.assertEqual(score_examples(rows,range(6),2),score_examples(list(reversed(rows)),range(6),2))

    def test_inconsistent_training_labels_are_rejected(self):
        rows=histories()
        rows[-1]['label']=0
        with self.assertRaisesRegex(ValueError,'Inconsistent'):
            score_examples(rows,range(6),2)

    def test_evaluation_checks_id_and_label_alignment(self):
        scores=score_examples(histories(),range(6),2)
        metadata=[dict(sample_id=i,label=i//3,background=(1-i//3) if i%3==2 else i//3) for i in range(6)]
        result=evaluate(scores,metadata,'class_relative_loss')
        self.assertEqual(result['selected_count'],1)
        self.assertEqual(result['precision_at_10pct'],1)
        metadata[0]['label']=1
        with self.assertRaisesRegex(ValueError,'disagree'):
            evaluate(scores,metadata,'mean_loss')


if __name__=='__main__':
    unittest.main()
