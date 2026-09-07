"""The independent corpus cache reuses parsing without hiding source edits."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import corpus


class CorpusCacheTests(unittest.TestCase):
    def test_content_changes_invalidate_parsing_even_with_unchanged_metadata(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "Example.lean"
            path.write_text("first")
            original = path.stat()

            def extract(filename, module):
                return [SimpleNamespace(name="example", py_src=None,
                                        body=Path(filename).read_text())]

            corpus._parse_leanexpr_table.cache_clear()
            with patch.object(corpus, "_all_modules", return_value=["Example"]), \
                    patch.object(corpus, "_module_path", return_value=str(path)), \
                    patch.object(corpus.L, "extract_file", side_effect=extract) as parser, \
                    patch.object(corpus.L, "extract_recursions", return_value=[]):
                _, first = corpus._leanexpr_table()
                first.clear()
                _, reused = corpus._leanexpr_table()
                self.assertEqual(reused["example"].body, "first")
                self.assertEqual(parser.call_count, 1)
                path.write_text("other")
                os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
                _, changed = corpus._leanexpr_table()
                self.assertEqual(changed["example"].body, "other")
                self.assertEqual(parser.call_count, 2)
            corpus._parse_leanexpr_table.cache_clear()


if __name__ == "__main__":
    unittest.main()
