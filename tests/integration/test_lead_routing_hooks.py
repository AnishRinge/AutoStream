import unittest

from agent.graph import build_graph
from agent.routing import ROUTE_BUSINESS, ROUTE_LEAD
from agent.state import (
    INTENT_BUSINESS_QUERY,
    INTENT_LEAD_INTEREST,
    INTENT_UNKNOWN,
    LEAD_STATE_AWAITING_NAME,
    LEAD_STATE_IDLE,
    TRANSIENT_STATE_KEYS,
)


def build_turn_state(user_input, persisted_state):
    turn_state = {**persisted_state, "input": user_input}
    for key in TRANSIENT_STATE_KEYS:
        if key != "input":
            turn_state.pop(key, None)
    return turn_state


def extract_persisted_state(result_state):
    return {k: v for k, v in result_state.items() if k not in TRANSIENT_STATE_KEYS}


class TestLeadRoutingHooks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = build_graph()

    def test_business_query_during_active_lead_preserves_state(self):
        persisted = {
            "lead_state": LEAD_STATE_AWAITING_NAME,
            "lead_name": None,
            "lead_email": None,
        }
        result = self.graph.invoke(
            build_turn_state("What is the refund policy?", persisted)
        )
        self.assertEqual(result["intent"], INTENT_BUSINESS_QUERY)
        self.assertEqual(result["route"], ROUTE_BUSINESS)
        self.assertIn("No refunds after 7 days", result["response"])

        next_state = extract_persisted_state(result)
        self.assertEqual(next_state.get("lead_state"), LEAD_STATE_AWAITING_NAME)
        self.assertIn("lead_name", next_state)
        self.assertIn("lead_email", next_state)

    def test_next_turn_collects_name_after_lead_start(self):
        graph = self.graph
        state = {"lead_state": LEAD_STATE_IDLE}
        started = graph.invoke(build_turn_state("I want to sign up", state))
        state = extract_persisted_state(started)
        self.assertEqual(state.get("lead_state"), LEAD_STATE_AWAITING_NAME)

        collected = graph.invoke(build_turn_state("Alex", state))
        self.assertEqual(collected["route"], ROUTE_LEAD)
        next_state = extract_persisted_state(collected)
        self.assertEqual(next_state.get("lead_name"), "Alex")
        self.assertEqual(next_state.get("lead_state"), "awaiting_email")

    def test_lead_interest_maps_to_dedicated_lead_route(self):
        result = self.graph.invoke({"input": "I want to sign up"})
        self.assertEqual(result["intent"], INTENT_LEAD_INTEREST)
        self.assertEqual(result["route"], ROUTE_LEAD)

    def test_idle_unknown_field_value_does_not_enter_lead(self):
        result = self.graph.invoke({"input": "Alex", "lead_state": LEAD_STATE_IDLE})
        self.assertNotEqual(result["route"], ROUTE_LEAD)


if __name__ == "__main__":
    unittest.main()
