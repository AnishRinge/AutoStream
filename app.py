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

from agent.db.lead_repository import LeadRepository, LeadRepositoryError
from agent.graph import build_graph
from agent.state import LEAD_STATE_CONFIRMED, TRANSIENT_STATE_KEYS

PROJECT_ROOT = Path(__file__).resolve().parent
graph = build_graph()


def get_lead_repository(db_path=None):
    if db_path is not None:
        return LeadRepository(db_path)
    # Prefer explicit env override (handled by repository); otherwise project-local default.
    import os

    if os.environ.get("LEADS_DB_PATH"):
        return LeadRepository(None)
    return LeadRepository(PROJECT_ROOT / "data" / "leads.db")


def _confirmed_lead_payload(state):
    return {
        "submission_id": state.get("lead_submission_id"),
        "name": state.get("lead_name"),
        "email": state.get("lead_email"),
        "platform": state.get("lead_platform"),
    }


def persist_confirmed_lead_if_needed(state, repository=None):
    """Persist a confirmed lead once. Safe/idempotent; preserves data on failure."""
    if state.get("lead_state") != LEAD_STATE_CONFIRMED:
        return state
    if state.get("lead_persisted") is True:
        return state

    payload = _confirmed_lead_payload(state)
    if not payload.get("submission_id") or not all(
        payload.get(k) for k in ("name", "email", "platform")
    ):
        return state

    repo = repository or get_lead_repository()
    try:
        result = repo.save_confirmed_lead(payload)
        if result.saved:
            state["lead_persisted"] = True
            state["response"] = "Thanks — your details are confirmed and saved."
    except LeadRepositoryError:
        state["lead_persisted"] = False
        state["response"] = (
            "Your details are confirmed, but I couldn't save them right now. "
            "Your information is still here — please ask me to retry saving when ready."
        )
    return state


def build_turn_state(user_input, persisted_state):
    """Create a fresh per-turn state without mutating persisted state."""
    turn_state = {**persisted_state, "input": user_input}
    # Ensure stale transient values never leak into the next turn.
    for key in TRANSIENT_STATE_KEYS:
        if key != "input":
            turn_state.pop(key, None)
    return turn_state


def update_conversation_history(state, user_input, response, max_entries=20):
    """Append one user/assistant turn; bounded to avoid unbounded duplication."""
    history = list(state.get("conversation_history") or [])
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": response or ""})
    if max_entries > 0 and len(history) > max_entries:
        history = history[-max_entries:]
    state["conversation_history"] = history
    return state


def extract_persisted_state(result_state):
    """Keep only durable conversation state across turns."""
    return {
        key: value
        for key, value in result_state.items()
        if key not in TRANSIENT_STATE_KEYS
    }


def main():
    state = {}
    print("AutoStream AI Agent (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        turn_state = build_turn_state(user_input, state)
        result = graph.invoke(turn_state)

        # Replace persisted state for the next turn instead of mutating it in-place.
        state = extract_persisted_state(result)
        persist_confirmed_lead_if_needed(state)
        update_conversation_history(state, user_input, state.get("response", ""))

        print("Agent:", state.get("response", ""))


if __name__ == "__main__":
    main()
