import unittest

from agent.kb.loader import clear_cache
from agent.kb.lookup import LookupStatus, lookup_business_query


class TestDirectLookup(unittest.TestCase):
    def tearDown(self):
        clear_cache()

    def test_pricing_all_plans(self):
        result = lookup_business_query("What plans do you offer?")
        self.assertEqual(result.status, LookupStatus.ANSWERED)
        self.assertIn("$29/month", result.response)
        self.assertIn("$79/month", result.response)
        self.assertIn("Basic Plan", result.response)

    def test_pro_price(self):
        result = lookup_business_query("How much does the Pro plan cost?")
        self.assertEqual(result.status, LookupStatus.ANSWERED)
        self.assertIn("$79/month", result.response)

    def test_basic_videos(self):
        result = lookup_business_query("How many videos can I make on the basic plan?")
        self.assertEqual(result.status, LookupStatus.ANSWERED)
        self.assertIn("10 videos", result.response)

    def test_refund_policy(self):
        result = lookup_business_query("What is your refund policy?")
        self.assertEqual(result.status, LookupStatus.ANSWERED)
        self.assertIn("No refunds after 7 days", result.response)

    def test_support_policy(self):
        result = lookup_business_query("How can I contact support?")
        self.assertEqual(result.status, LookupStatus.ANSWERED)
        self.assertIn("support@autostream.com", result.response)
        self.assertIn("24/7", result.response)

    def test_recognized_business_without_verified_fact(self):
        result = lookup_business_query("What integrations do you support?")
        self.assertEqual(result.status, LookupStatus.NOT_FOUND)
        self.assertIn("verified", result.response.lower())

    def test_complex_question_deferred_for_rag(self):
        result = lookup_business_query("Which plan is better for a small YouTube creator?")
        self.assertEqual(result.status, LookupStatus.DEFER_RAG)

    def test_no_fabrication(self):
        result = lookup_business_query("Do you offer enterprise pricing?")
        self.assertNotIn("$999", result.response)
        self.assertNotIn("custom quote available now", result.response.lower())


if __name__ == "__main__":
    unittest.main()
