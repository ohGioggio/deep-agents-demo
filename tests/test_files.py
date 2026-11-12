"""Files the agent delivers: validation, the send_file tool, events and chat commands."""
import unittest

from deep_agents_demo.features.files.validate import clean_name, describe, validate

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
JPG = b"\xff\xd8\xff" + b"\x00" * 16


class ValidationTests(unittest.TestCase):
    def assertRefused(self, name, data, *words):
        error = validate(name, data)
        self.assertIsNotNone(error, f"{name} should be refused")
        for word in words:
            self.assertIn(word, error)

    def test_valid_files(self):
        for name, data in [("a.md", b"# Title\n"), ("a.txt", b"note"), ("a.tex", b"\\section{x}"),
                           ("a.json", b'{"a": 1}'), ("a.csv", b"a,b\n1,2\n"), ("a.html", b"<p>hi</p>"),
                           ("a.svg", b'<svg xmlns="http://www.w3.org/2000/svg"></svg>'),
                           ("a.png", PNG), ("a.jpg", JPG), ("a.jpeg", JPG)]:
            with self.subTest(name=name):
                self.assertIsNone(validate(name, data))

    def test_refusals(self):
        self.assertRefused("report.docx", b"x", ".md", ".csv")
        self.assertRefused("a.md", b"", "empty")
        self.assertRefused("a.md", b"\xff\xfe\xfa", "UTF-8")
        self.assertRefused("a.md", b"x" * (5 * 1024 * 1024 + 1), "5 MB")
        self.assertRefused("a.png", PNG + b"x" * (20 * 1024 * 1024), "20 MB")
        self.assertRefused("a.csv", b"a,b\n1\n", "column")
        self.assertRefused("a.json", b"{", "JSON")
        self.assertRefused("a.svg", b"<html/>", "<svg>")
        self.assertRefused("a.svg", b"<svg", "SVG")
        self.assertRefused("a.html", b"just text", "HTML")
        self.assertRefused("a.png", JPG, "PNG")
        self.assertRefused("a.jpg", b"not an image", "JPEG")
        self.assertRefused("keys.md", b"key: AKIAABCDEFGHIJKLMNOP", "credential")

    def test_names_are_sanitised(self):
        self.assertEqual(clean_name("/sandbox/Report Final.MD"), "Report-Final.md")
        self.assertEqual(clean_name("..\\..\\x.md"), "x.md")
        self.assertEqual(clean_name("C:/a/b.Csv"), "b.csv")
        for bad in (".md", "noext", "..", "dir/"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                clean_name(bad)

    def test_describe(self):
        self.assertEqual(describe("t.csv", b"a,b\n1,2\n3,4\n"), "csv, 3 rows x 2 columns")
        self.assertEqual(describe("a.md", b"x" * 1300), "md, 1.3 KB")


if __name__ == "__main__":
    unittest.main()
