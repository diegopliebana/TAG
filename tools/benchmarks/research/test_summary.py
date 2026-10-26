"""Check process-level statistics against hand-computed examples."""
import unittest
from summarize import aggregate

class ProcessStatistics(unittest.TestCase):
    def row(self,fork,values):
        return dict(engine='rust',game='fixture',players=2,operation='copy',fork=fork,samples=[dict(calls=2,elapsed_ns=2*x) for x in values])

    def test_process_medians_are_not_pooled_batch_median(self):
        # Process medians are 3, 8 and 100. Pooling all 15 batches gives 6.
        rows=[self.row(1,[1,2,3,4,5]),self.row(2,[6,7,8,9,10]),self.row(3,[0.1,0.1,100,100,100])]
        result,=aggregate(rows)
        self.assertEqual(result['median_ns'],8)
        self.assertEqual((result['min_fork_ns'],result['max_fork_ns']),(3,100))

    def test_elapsed_time_is_normalized_by_completed_calls(self):
        result,=aggregate([self.row(i,[10]*5) for i in [1,2,3]])
        self.assertEqual(result['median_ns'],10)

    def test_repeated_batches_do_not_replace_independent_processes(self):
        with self.assertRaises(AssertionError):aggregate([self.row(1,[10]*5)]*3)

if __name__=='__main__':unittest.main()
