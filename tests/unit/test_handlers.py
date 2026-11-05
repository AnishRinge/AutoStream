import unittest

from agent.handlers.business import handle_business
from agent.handlers.fallback import handle_fallback
from agent.handlers.greeting import handle_greeting
from agent.state import INTENT_GOODBYE, INTENT_LEAD_INTEREST, INTENT_THANKS


class TestHandlers(unittest.TestCase):
    def test_greeting(self):
        text = handle_greeting({"intent": "greeting"})
        self.assertIn("AutoStream", text)

    def test_thanks(self):
        text = handle_greeting({"intent": INTENT_THANKS})
        self.assertIn("welcome", text.lower())

    def test_goodbye(self):
        text = handle_greeting({"intent": INTENT_GOODBYE})
        self.assertIn("goodbye", text.lower())

    def test_business_generic_message(self):
        text = handle_business({"intent": "business_query", "input": "Tell me about AutoStream"})
        self.assertIn("AutoStream", text)

    def test_business_factual_lookup(self):
        text = handle_business({"intent": "business_query", "input": "What is your refund policy?"})
        self.assertIn("No refunds after 7 days", text)

    def test_business_unsupported_question(self):
        text = handle_business({"intent": "business_query", "input": "What integrations do you support?"})
        self.assertIn("verified", text.lower())

    def test_lead_interest_starts_collection(self):
        text = handle_business({"intent": INTENT_LEAD_INTEREST, "lead_state": "idle"})
        self.assertIn("name", text.lower())

    def test_fallback_no_fabrication(self):
        text = handle_fallback({"intent": "unknown"})
        self.assertIn("rephrase", text.lower())
        self.assertNotIn("$29", text)
        self.assertNotIn("refund after 30", text.lower())


if __name__ == "__main__":
    unittest.main()
