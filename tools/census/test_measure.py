"""Boundary checks for the physical-file census."""

import unittest
from measure import category, line_counts, public, summarize
from test_inventory import rust_inventory


class CensusTests(unittest.TestCase):
    def test_shared_runner_tests_merge_but_crates_remain_distinct(self):
        records = rust_inventory("\n".join([
            "Running tests (target/debug/tag_rust-123.exe)",
            "test tests::same_name ... ok",
            "test matches::tests::reused ... ok",
            "Running tests (target/debug/tag_browser-123.exe)",
            "test tests::same_name ... ok",
            "test matches::tests::reused ... ok",
        ]))
        self.assertEqual(len(records), 3)
        self.assertEqual(sum(len(r["executions"]) for r in records), 4)

    def test_physical_lines(self):
        self.assertEqual(line_counts(b"\xef\xbb\xbf// comment\r\n\r\nlast"), (3, 1, 2))
        self.assertEqual(line_counts(b""), (0, 0, 0))
        self.assertEqual(line_counts(b"x\n"), (1, 0, 1))

    def test_scope_and_overlapping_roles(self):
        self.assertFalse(public("docs/project/progress.md"))
        self.assertTrue(public("docs/reference/framework/overview.md"))
        self.assertEqual(category("examples/selfplay/results/model.pt"), "Training outputs")
        self.assertEqual(category("tools/java-comparison/BenchmarkResearch.java"), "Benchmark tools")
        self.assertEqual(category("adapters/python/tests/test_neural.py"), "Tests")

    def test_totals_preserve_binary_files_without_inventing_lines(self):
        result = summarize([dict(category="Assets", language="Binary PNG", bytes=12,
                                 lines=None, blank=None, nonblank=None)])
        self.assertEqual(result[0]["files"], 1)
        self.assertEqual(result[0]["bytes"], 12)
        self.assertEqual(result[0]["lines"], 0)


if __name__ == "__main__":
    unittest.main()
