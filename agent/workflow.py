from agent.intent import detect_intent
from agent.rag import get_relevant_info
from agent.tools import mock_lead_capture

class AgentState:
    def __init__(self):
        self.name = None
        self.email = None
        self.platform = None
        self.is_high_intent = False

state = AgentState()

def handle_query(user_input):
    query = user_input.lower()
    intent = detect_intent(user_input)

    # if user is already in lead flow, collect details first
    if state.is_high_intent:
        if state.name is None:
            state.name = user_input
            return "Thanks! Now please provide your email."

        if state.email is None:
            state.email = user_input
            return "Got it! Which platform do you create content on? (YouTube, Instagram, etc.)"

        if state.platform is None:
            state.platform = user_input

            mock_lead_capture(state.name, state.email, state.platform)

            state.name = None
            state.email = None
            state.platform = None
            state.is_high_intent = False

            return "Awesome! You’re all set. Our team will reach out to you soon."

    # answer policy questions first if explicitly asked
    if any(word in query for word in ["policy", "policies", "refund", "support"]):
        data = get_relevant_info(user_input)

        if isinstance(data, dict) and "refund" in data and "support" in data:
            return (
                "Here are our company policies:\n\n"
                f"- {data['refund']}\n"
                f"- {data['support']}"
            )

        return "Sorry, I couldn't find policy details."

    # then handle pricing queries
    if intent != "high_intent" and any(word in query for word in ["price", "pricing", "plan", "cost"]):
        data = get_relevant_info(user_input)

        if isinstance(data, dict) and "basic" in data:
            return (
                "Here are our pricing plans:\n\n"
                "Basic Plan: $29/month\n"
                "- 10 videos/month\n"
                "- 720p resolution\n\n"
                "Pro Plan: $79/month\n"
                "- Unlimited videos\n"
                "- 4K resolution\n"
                "- AI captions"
            )

        return "Sorry, I couldn't find pricing details."

    # then check if user wants to proceed
    if intent == "high_intent":
        state.is_high_intent = True
        return "Great! I can help you get started. Could you please share your name?"

    # greeting
    if intent == "greeting":
        return "Hello! How can I help you today?"

    # simple thanks handling (only if nothing else matched)
    if "thank" in query:
        return "You're welcome! Let me know if you have any other questions."

    return "I’m not sure how to help with that. Could you please clarify?"