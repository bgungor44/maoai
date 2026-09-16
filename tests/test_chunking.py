import unittest

from chunking import create_chunks


class ChunkingTests(unittest.TestCase):
    def test_empty_text(self):
        self.assertEqual(create_chunks(""), [])

    def test_keeps_small_paragraphs_together(self):
        text = "Birinci paragraf.\n\nİkinci paragraf."
        chunks = create_chunks(text, chunk_size=100, overlap=10)
        self.assertEqual(len(chunks), 1)
        self.assertIn("Birinci paragraf", chunks[0])
        self.assertIn("İkinci paragraf", chunks[0])

    def test_splits_long_paragraph(self):
        text = "A" * 250
        chunks = create_chunks(text, chunk_size=100, overlap=20)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 100 for chunk in chunks))

    def test_ignores_blank_paragraphs(self):
        text = "\n\nİçerik\n\n\n\n"
        chunks = create_chunks(text, chunk_size=100, overlap=10)
        self.assertEqual(chunks, ["İçerik"])


if __name__ == "__main__":
    unittest.main()
