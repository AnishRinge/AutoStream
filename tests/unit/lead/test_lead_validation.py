import unittest

from agent.handlers.lead_validation import validate_email, validate_name, validate_platform


class TestLeadValidation(unittest.TestCase):
    def test_validate_name(self):
        self.assertTrue(validate_name("Alex Rivera").valid)
        self.assertTrue(validate_name("O'Neil").valid)
        self.assertFalse(validate_name("A").valid)
        self.assertFalse(validate_name("12345").valid)
        self.assertFalse(validate_name("").valid)

    def test_validate_email(self):
        self.assertTrue(validate_email("alex@example.com").valid)
        self.assertTrue(validate_email("alex+pro@sub.example.co").valid)
        self.assertFalse(validate_email("alex@example").valid)
        self.assertFalse(validate_email("alexexample.com").valid)
        self.assertFalse(validate_email("alex@@example.com").valid)
        self.assertFalse(validate_email("").valid)

    def test_validate_platform(self):
        self.assertTrue(validate_platform("YouTube").valid)
        self.assertTrue(validate_platform("instagram").valid)
        self.assertTrue(validate_platform("twitch live").valid)
        self.assertFalse(validate_platform("x").valid)
        self.assertFalse(validate_platform("!!!").valid)
        self.assertFalse(validate_platform("").valid)
