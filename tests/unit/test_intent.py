import unittest

from agent.intent import detect_intent
from agent.state import (
    INTENT_BUSINESS_QUERY,
    INTENT_CANCEL,
    INTENT_CONFIRM,
    INTENT_GOODBYE,
    INTENT_GREETING,
    INTENT_LEAD_INTEREST,
    INTENT_THANKS,
    INTENT_UNKNOWN,
)


class TestIntentClassification(unittest.TestCase):
    def test_greetings(self):
        for text in ["hi", "Hello", "hey there", "Good morning"]:
            with self.subTest(text=text):
                self.assertEqual(detect_intent(text), INTENT_GREETING)

    def test_thanks(self):
        for text in ["thanks", "thank you", "thx!", "much appreciated"]:
            with self.subTest(text=text):
                self.assertEqual(detect_intent(text), INTENT_THANKS)

    def test_goodbye(self):
        for text in ["bye", "goodbye", "see you later", "good night"]:
            with self.subTest(text=text):
                self.assertEqual(detect_intent(text), INTENT_GOODBYE)

    def test_lead_interest(self):
        for text in [
            "I want to try the Pro plan",
            "sign me up",
            "I want to buy the basic plan",
            "I'd like to purchase",
            "ready to proceed",
            "interested in subscribing",  # contains subscribe
            "I want to purchase it",
        ]:
            with self.subTest(text=text):
                self.assertEqual(detect_intent(text), INTENT_LEAD_INTEREST)

    def test_business_queries_not_misclassified_as_lead(self):
        for text in [
            "What are your pricing plans?",
            "Tell me about the Pro plan",
            "Do you have a refund policy?",
            "I'm interested in pricing",
            "What features are included?",
        ]:
            with self.subTest(text=text):
                self.assertEqual(detect_intent(text), INTENT_BUSINESS_QUERY)

    def test_explicit_cancel_confirm_labels(self):
        self.assertEqual(detect_intent("cancel"), INTENT_CANCEL)
        self.assertEqual(detect_intent("never mind"), INTENT_CANCEL)
        self.assertEqual(detect_intent("yes, confirm"), INTENT_CONFIRM)
        self.assertEqual(detect_intent("looks good"), INTENT_CONFIRM)

    def test_unknown(self):
        for text in ["asdfgh", "tell me a joke", ""]:
            with self.subTest(text=text):
                self.assertEqual(detect_intent(text), INTENT_UNKNOWN)


if __name__ == "__main__":
    unittest.main()
