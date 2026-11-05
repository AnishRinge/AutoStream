import unittest

import app
from agent.graph import build_graph
from agent.state import LEAD_STATE_IDLE, TRANSIENT_STATE_KEYS


def build_turn_state(user_input, persisted_state):
    """Mirror app.py turn-state behavior for CLI-style tests."""
    return app.build_turn_state(user_input, persisted_state)


def extract_persisted_state(result_state):
    return app.extract_persisted_state(result_state)


class TestStatePersistence(unittest.TestCase):
    def test_fresh_input_does_not_overwrite_persisted_fields(self):
        graph = build_graph()
        persisted = {"lead_state": LEAD_STATE_IDLE, "lead_name": "Alex"}
        result = graph.invoke(build_turn_state("hello", persisted))
        next_state = extract_persisted_state(result)

        self.assertEqual(next_state.get("lead_name"), "Alex")
        self.assertEqual(next_state.get("lead_state"), LEAD_STATE_IDLE)
        self.assertNotIn("input", next_state)
        self.assertNotIn("response", next_state)
        self.assertNotIn("intent", next_state)

    def test_consecutive_turns_keep_durable_state(self):
        graph = build_graph()
        state = {"lead_name": "Sam"}
        for user_text in ["hi", "what are your plans?", "thanks"]:
            result = graph.invoke(build_turn_state(user_text, state))
            state = extract_persisted_state(result)
            app.update_conversation_history(state, user_text, result.get("response"))
            self.assertEqual(state.get("lead_name"), "Sam")
            self.assertTrue(result.get("response"))

    def test_transient_keys_are_reset_between_turns(self):
        graph = build_graph()
        state = {}
        first = graph.invoke(build_turn_state("hello", state))
        state = extract_persisted_state(first)
        app.update_conversation_history(state, "hello", first.get("response"))
        second = graph.invoke(build_turn_state("tell me a joke", state))
        self.assertNotEqual(first.get("intent"), second.get("intent"))
        self.assertNotEqual(first.get("response"), second.get("response"))

    def test_conversation_history_appends_once_per_turn_and_is_bounded(self):
        graph = build_graph()
        state = {}
        for i in range(12):
            text = f"hello {i}"
            result = graph.invoke(build_turn_state(text, state))
            state = extract_persisted_state(result)
            app.update_conversation_history(state, text, result.get("response"), max_entries=10)

        history = state.get("conversation_history", [])
        self.assertEqual(len(history), 10)
        self.assertEqual(history[-2]["role"], "user")
        self.assertEqual(history[-2]["content"], "hello 11")
        self.assertEqual(history[-1]["role"], "assistant")

    def test_current_input_separate_from_history(self):
        graph = build_graph()
        state = {"conversation_history": [{"role": "user", "content": "old"}]}
        turn = build_turn_state("new input", state)
        self.assertEqual(turn["input"], "new input")
        self.assertEqual(turn["conversation_history"][0]["content"], "old")
        result = graph.invoke(turn)
        persisted = extract_persisted_state(result)
        self.assertNotIn("input", persisted)


if __name__ == "__main__":
    unittest.main()
