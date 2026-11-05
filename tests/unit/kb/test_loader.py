import json
import tempfile
import unittest
from pathlib import Path

from agent.kb.loader import KnowledgeBaseError, clear_cache, load_knowledge


VALID_KB = {
    "pricing": {
        "basic": {"price": "$29/month", "features": ["10 videos per month", "720p resolution"]},
        "pro": {
            "price": "$79/month",
            "features": ["Unlimited videos", "4K resolution", "AI captions"],
        },
    },
    "policies": {
        "refund": "No refunds after 7 days",
        "support": "Pro Plan customers receive 24/7 support via email at support@autostream.com.",
    },
}


class TestKnowledgeBaseLoader(unittest.TestCase):
    def setUp(self):
        clear_cache()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(clear_cache)

    def _write(self, name, content):
        path = Path(self.tmp.name) / name
        path.write_text(content, encoding="utf-8")
        return path

    def test_loads_valid_json(self):
        path = self._write("kb.json", json.dumps(VALID_KB))
        data = load_knowledge(path)
        self.assertEqual(data["pricing"]["pro"]["price"], "$79/month")
        self.assertEqual(data["policies"]["refund"], "No refunds after 7 days")

    def test_missing_file(self):
        with self.assertRaises(KnowledgeBaseError):
            load_knowledge(Path(self.tmp.name) / "missing.json")

    def test_malformed_json(self):
        path = self._write("bad.json", "{not json")
        with self.assertRaises(KnowledgeBaseError):
            load_knowledge(path)

    def test_unexpected_structure(self):
        path = self._write("bad.json", json.dumps({"pricing": "nope"}))
        with self.assertRaises(KnowledgeBaseError):
            load_knowledge(path)

    def test_cache_and_force_reload(self):
        path = self._write("kb.json", json.dumps(VALID_KB))
        first = load_knowledge(path)
        second = load_knowledge(path)
        self.assertEqual(first, second)
        path.write_text(
            json.dumps(
                {
                    **VALID_KB,
                    "policies": {**VALID_KB["policies"], "refund": "Changed"},
                }
            ),
            encoding="utf-8",
        )
        # Force reload to observe file changes in this test.
        refreshed = load_knowledge(path, force_reload=True)
        self.assertEqual(refreshed["policies"]["refund"], "Changed")

    def test_returns_copy_not_cached_mutation(self):
        path = self._write("kb.json", json.dumps(VALID_KB))
        data = load_knowledge(path)
        data["policies"]["refund"] = "mutated"
        again = load_knowledge(path)
        self.assertEqual(again["policies"]["refund"], "No refunds after 7 days")


if __name__ == "__main__":
    unittest.main()
