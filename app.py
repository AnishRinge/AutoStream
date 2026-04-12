import sys
from pathlib import Path


def bootstrap_local_venv():
    project_root = Path(__file__).resolve().parent
    venv_site_packages = project_root / "venv" / "Lib" / "site-packages"

    if venv_site_packages.exists():
        site_packages_path = str(venv_site_packages)
        if site_packages_path not in sys.path:
            sys.path.insert(0, site_packages_path)


bootstrap_local_venv()

from dotenv import load_dotenv

load_dotenv()

from agent.graph import build_graph

graph = build_graph()

print("AutoStream AI Agent (type 'exit' to quit)\n")

TRANSIENT_STATE_KEYS = {"input", "response", "intent"}


def build_turn_state(user_input, persisted_state):
    # Always create a fresh state object for this turn so old values
    # cannot override the latest user input.
    return {**persisted_state, "input": user_input}


def extract_persisted_state(result_state):
    # Keep only long-lived conversation fields between turns.
    return {
        key: value
        for key, value in result_state.items()
        if key not in TRANSIENT_STATE_KEYS
    }


state = {}

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    turn_state = build_turn_state(user_input, state)
    result = graph.invoke(turn_state)

    # Replace persisted state for the next turn instead of mutating it in-place.
    state = extract_persisted_state(result)

    print("Agent:", result["response"])
