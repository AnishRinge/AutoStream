import unittest

from agent.kb.retrieval.records import KnowledgeRecord, build_knowledge_records, records_by_id


class TestKnowledgeRecords(unittest.TestCase):
    def setUp(self):
        self.kb = {
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

    def test_builds_verified_records_with_stable_ids(self):
        records = build_knowledge_records(self.kb)
        ids = [r.id for r in records]
        self.assertIn("pricing.basic", ids)
        self.assertIn("pricing.pro", ids)
        self.assertIn("policy.refund", ids)
        self.assertIn("policy.support", ids)
        self.assertIn("pricing.basic.feature.0", ids)
        self.assertTrue(all(r.source == "knowledge_base/data.json" for r in records))
        self.assertTrue(all(r.text for r in records))

    def test_preserves_verified_facts_in_text_and_metadata(self):
        by_id = records_by_id(build_knowledge_records(self.kb))
        refund = by_id["policy.refund"]
        self.assertIn("No refunds after 7 days", refund.text)
        self.assertEqual(refund.metadata["value"], "No refunds after 7 days")
        pro = by_id["pricing.pro"]
        self.assertEqual(pro.metadata["price"], "$79/month")
        self.assertIn("AI captions", pro.text)

    def test_empty_kb_returns_empty(self):
        self.assertEqual(build_knowledge_records({}), [])


if __name__ == "__main__":
    unittest.main()
