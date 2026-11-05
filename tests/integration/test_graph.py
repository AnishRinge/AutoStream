import unittest

from agent.graph import build_graph
from agent.routing import ROUTE_BUSINESS, ROUTE_FALLBACK, ROUTE_GREETING, ROUTE_LEAD
from agent.state import (
    INTENT_BUSINESS_QUERY,
    INTENT_GOODBYE,
    INTENT_GREETING,
    INTENT_LEAD_INTEREST,
    INTENT_THANKS,
    INTENT_UNKNOWN,
)


class TestGraphIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = build_graph()

    def _invoke(self, text):
        return self.graph.invoke({"input": text})

    def test_greeting_flow(self):
        result = self._invoke("hello")
        self.assertEqual(result["intent"], INTENT_GREETING)
        self.assertEqual(result["route"], ROUTE_GREETING)
        self.assertIn("AutoStream", result["response"])

    def test_thanks_flow(self):
        result = self._invoke("thank you")
        self.assertEqual(result["intent"], INTENT_THANKS)
        self.assertEqual(result["route"], ROUTE_GREETING)

    def test_goodbye_flow(self):
        result = self._invoke("bye")
        self.assertEqual(result["intent"], INTENT_GOODBYE)
        self.assertEqual(result["route"], ROUTE_GREETING)

    def test_business_query_flow(self):
        result = self._invoke("What are your pricing plans?")
        self.assertEqual(result["intent"], INTENT_BUSINESS_QUERY)
        self.assertEqual(result["route"], ROUTE_BUSINESS)
        self.assertIn("$29/month", result["response"])
        self.assertIn("$79/month", result["response"])

    def test_lead_interest_flow(self):
        result = self._invoke("I want to try the Pro plan")
        self.assertEqual(result["intent"], INTENT_LEAD_INTEREST)
        self.assertEqual(result["route"], ROUTE_LEAD)
        self.assertTrue(result["response"])

    def test_unknown_flow(self):
        result = self._invoke("tell me a joke")
        self.assertEqual(result["intent"], INTENT_UNKNOWN)
        self.assertEqual(result["route"], ROUTE_FALLBACK)
        self.assertIn("rephrase", result["response"].lower())


if __name__ == "__main__":
    unittest.main()
