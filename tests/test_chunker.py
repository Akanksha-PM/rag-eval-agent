"""Tests for src.ingestion.chunker."""

import unittest
from unittest.mock import patch

from config import Config
from src.ingestion import chunker


class ChunkTextTestCase(unittest.TestCase):
    def test_short_text_returns_single_chunk(self):
        text = "A short paragraph that easily fits inside one chunk."
        with patch.object(Config, "CHUNK_SIZE", 50), patch.object(Config, "CHUNK_OVERLAP", 5):
            chunks = chunker.chunk_text(text, "Test Product", "source.txt")

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["text"], text)
        self.assertEqual(chunks[0]["word_count"], len(text.split()))

    def test_empty_text_returns_empty_list(self):
        with patch.object(Config, "CHUNK_SIZE", 50), patch.object(Config, "CHUNK_OVERLAP", 5):
            self.assertEqual(chunker.chunk_text("", "Test Product", "source.txt"), [])
            self.assertEqual(
                chunker.chunk_text("   \n\n  ", "Test Product", "source.txt"), []
            )

    def test_long_text_produces_multiple_chunks_none_over_limit(self):
        # 5 paragraphs of 6 distinct words each; chunk_size leaves headroom
        # for the overlap so it isn't capped by the size ceiling.
        paragraphs = [" ".join(f"p{p}w{w}" for w in range(6)) for p in range(5)]
        text = "\n\n".join(paragraphs)

        with patch.object(Config, "CHUNK_SIZE", 15), patch.object(Config, "CHUNK_OVERLAP", 3):
            chunks = chunker.chunk_text(text, "Test Product", "source.txt")

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(chunk["word_count"], 15)

    def test_consecutive_chunks_share_overlapping_words(self):
        paragraphs = [" ".join(f"p{p}w{w}" for w in range(6)) for p in range(5)]
        text = "\n\n".join(paragraphs)

        with patch.object(Config, "CHUNK_SIZE", 15), patch.object(Config, "CHUNK_OVERLAP", 3):
            chunks = chunker.chunk_text(text, "Test Product", "source.txt")

        self.assertGreater(len(chunks), 1)
        for prev_chunk, next_chunk in zip(chunks, chunks[1:]):
            prev_words = prev_chunk["text"].split()
            next_words = next_chunk["text"].split()
            self.assertEqual(prev_words[-3:], next_words[:3])

    def test_paragraph_longer_than_chunk_size_is_hard_split(self):
        words = [f"word{i}" for i in range(12)]
        text = " ".join(words)  # single paragraph, no blank lines

        with patch.object(Config, "CHUNK_SIZE", 5), patch.object(Config, "CHUNK_OVERLAP", 0):
            chunks = chunker.chunk_text(text, "Test Product", "source.txt")

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0]["text"], " ".join(words[0:5]))
        self.assertEqual(chunks[1]["text"], " ".join(words[5:10]))
        self.assertEqual(chunks[2]["text"], " ".join(words[10:12]))
        for chunk in chunks:
            self.assertLessEqual(chunk["word_count"], 5)

    def test_chunk_ids_and_metadata_set_correctly(self):
        text = "one two three four five six seven eight"
        with patch.object(Config, "CHUNK_SIZE", 3), patch.object(Config, "CHUNK_OVERLAP", 1):
            chunks = chunker.chunk_text(text, "Test Product", "https://example.com/docs")

        self.assertGreater(len(chunks), 1)
        for index, chunk in enumerate(chunks):
            self.assertEqual(chunk["id"], f"test_product_{index}")
            self.assertEqual(chunk["product"], "Test Product")
            self.assertEqual(chunk["source"], "https://example.com/docs")
            self.assertEqual(chunk["chunk_index"], index)


if __name__ == "__main__":
    unittest.main()
