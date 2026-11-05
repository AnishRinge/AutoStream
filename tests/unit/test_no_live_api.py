import builtins
import unittest

import agent.graph


class TestNoLiveApiCalls(unittest.TestCase):
    """Guardrail: Phase 1 tests must not perform live Gemini/LLM calls."""

    def test_graph_module_has_no_gemini_client(self):
        # Phase 1 graph uses deterministic handlers only.
        self.assertFalse(hasattr(agent.graph, "model"))
        self.assertFalse(hasattr(agent.graph, "generate_grounded_response"))
        self.assertFalse(hasattr(agent.graph, "genai"))

    def test_build_graph_import_does_not_require_api_key(self):
        # Building/invoking the graph must not configure or call external APIs.
        graph = agent.graph.build_graph()
        result = graph.invoke({"input": "hello"})
        self.assertIn("response", result)
        self.assertTrue(result["response"])


if __name__ == "__main__":
    unittest.main()
