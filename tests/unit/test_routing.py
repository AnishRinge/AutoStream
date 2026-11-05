import unittest

from agent.routing import (
    ROUTE_BUSINESS,
    ROUTE_FALLBACK,
    ROUTE_GREETING,
    ROUTE_LEAD,
    resolve_route,
    route_from_state,
)
from agent.state import (
    INTENT_BUSINESS_QUERY,
    INTENT_CANCEL,
    INTENT_CONFIRM,
    INTENT_GOODBYE,
    INTENT_GREETING,
    INTENT_LEAD_INTEREST,
    INTENT_THANKS,
    INTENT_UNKNOWN,
    LEAD_STATE_AWAITING_EMAIL,
    LEAD_STATE_AWAITING_NAME,
    LEAD_STATE_IDLE,
)


class TestRouting(unittest.TestCase):
    def test_intent_routes(self):
        cases = [
            (INTENT_GREETING, LEAD_STATE_IDLE, ROUTE_GREETING),
            (INTENT_THANKS, LEAD_STATE_IDLE, ROUTE_GREETING),
            (INTENT_GOODBYE, LEAD_STATE_IDLE, ROUTE_GREETING),
            (INTENT_BUSINESS_QUERY, LEAD_STATE_IDLE, ROUTE_BUSINESS),
            (INTENT_LEAD_INTEREST, LEAD_STATE_IDLE, ROUTE_LEAD),
            (INTENT_UNKNOWN, LEAD_STATE_IDLE, ROUTE_FALLBACK),
            (INTENT_CANCEL, LEAD_STATE_IDLE, ROUTE_FALLBACK),
            (INTENT_CONFIRM, LEAD_STATE_IDLE, ROUTE_FALLBACK),
            (INTENT_LEAD_INTEREST, "confirmed", ROUTE_LEAD),
            (INTENT_LEAD_INTEREST, "cancelled", ROUTE_LEAD),
        ]
        for intent, lead_state, expected in cases:
            with self.subTest(intent=intent):
                self.assertEqual(resolve_route(intent, lead_state), expected)

    def test_active_lead_state_routes_field_input_to_lead_hook(self):
        self.assertEqual(
            resolve_route(INTENT_UNKNOWN, LEAD_STATE_AWAITING_EMAIL),
            ROUTE_LEAD,
        )

    def test_business_query_interrupts_active_lead_to_business(self):
        self.assertEqual(
            resolve_route(INTENT_BUSINESS_QUERY, LEAD_STATE_AWAITING_NAME),
            ROUTE_BUSINESS,
        )

    def test_cancel_confirm_distinguishable_during_active_lead(self):
        self.assertEqual(resolve_route(INTENT_CANCEL, LEAD_STATE_AWAITING_NAME), ROUTE_LEAD)
        self.assertEqual(resolve_route(INTENT_CONFIRM, LEAD_STATE_AWAITING_EMAIL), ROUTE_LEAD)

    def test_route_from_state(self):
        self.assertEqual(
            route_from_state({"intent": INTENT_BUSINESS_QUERY, "lead_state": LEAD_STATE_IDLE}),
            ROUTE_BUSINESS,
        )


if __name__ == "__main__":
    unittest.main()
