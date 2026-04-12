import os

import google.generativeai as genai
from langgraph.graph import END, StateGraph
from typing_extensions import NotRequired, TypedDict

from agent.intent import detect_intent
from agent.rag import get_relevant_info
from agent.tools import mock_lead_capture


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")


class AgentState(TypedDict):
    input: str
    response: NotRequired[str]
    intent: NotRequired[str]
    lead_stage: NotRequired[str | None]
    name: NotRequired[str | None]
    email: NotRequired[str | None]
    platform: NotRequired[str | None]


def generate_grounded_response(prompt, fallback_text):
    try:
        response = model.generate_content(
            prompt,
            request_options={"timeout": 5},
        )
        text = getattr(response, "text", "").strip()
        return text or fallback_text
    except Exception:
        return fallback_text


def intent_node(state):
    state["intent"] = detect_intent(state["input"])
    return state


def policy_node(state):
    data = get_relevant_info(state["input"])

    if isinstance(data, dict) and "refund" in data and "support" in data:
        fallback_text = (
            "Here are our company policies:\n\n"
            f"- {data['refund']}\n"
            f"- {data['support']}"
        )
        prompt = (
            "You are an assistant for AutoStream, a SaaS video editing product.\n"
            "Answer the user's policy question using only the policy facts below.\n"
            "Do not invent any details.\n\n"
            f"User question: {state['input']}\n"
            f"Refund policy: {data['refund']}\n"
            f"Support policy: {data['support']}\n"
        )
        state["response"] = generate_grounded_response(prompt, fallback_text)
    else:
        state["response"] = "Sorry, I couldn't find policy details."

    return state


def pricing_node(state):
    data = get_relevant_info(state["input"])

    if isinstance(data, dict) and "basic" in data and "pro" in data:
        fallback_text = (
            "Here are our pricing plans:\n\n"
            "Basic Plan: $29/month\n"
            "- 10 videos/month\n"
            "- 720p resolution\n\n"
            "Pro Plan: $79/month\n"
            "- Unlimited videos\n"
            "- 4K resolution\n"
            "- AI captions"
        )
        prompt = (
            "You are an assistant for AutoStream, a SaaS video editing product.\n"
            "Answer the user's pricing question using only the pricing facts below.\n"
            "Keep it concise, helpful, and grounded in the provided data.\n"
            "Do not invent discounts, trials, or extra features.\n\n"
            f"User question: {state['input']}\n"
            f"Basic plan price: {data['basic']['price']}\n"
            f"Basic plan features: {', '.join(data['basic']['features'])}\n"
            f"Pro plan price: {data['pro']['price']}\n"
            f"Pro plan features: {', '.join(data['pro']['features'])}\n"
        )
        state["response"] = generate_grounded_response(prompt, fallback_text)
    else:
        state["response"] = "Sorry, I couldn't find pricing details."

    return state


def lead_node(state):
    state["response"] = "Great! I can help you get started. Could you please share your name?"
    state["lead_stage"] = "name"
    return state


def lead_collect_node(state):
    user_input = state["input"]
    lead_stage = state.get("lead_stage")

    if lead_stage == "name":
        state["name"] = user_input
        state["lead_stage"] = "email"
        state["response"] = "Thanks! Now please provide your email."
        return state

    if lead_stage == "email":
        state["email"] = user_input
        state["lead_stage"] = "platform"
        state["response"] = "Got it! Which platform do you create content on? (YouTube, Instagram, etc.)"
        return state

    if lead_stage == "platform":
        state["platform"] = user_input
        mock_lead_capture(state["name"], state["email"], state["platform"])
        state["response"] = "Awesome! You're all set. Our team will reach out to you soon."
        state["lead_stage"] = None
        state["name"] = None
        state["email"] = None
        state["platform"] = None
        return state

    state["response"] = "Great! I can help you get started. Could you please share your name?"
    state["lead_stage"] = "name"
    return state


def greeting_node(state):
    state["response"] = "Hello! How can I help you today?"
    return state


def fallback_node(state):
    query = state["input"].lower()

    if "thank" in query:
        state["response"] = "You're welcome! Let me know if you have any other questions."
    else:
        fallback_text = "I'm not sure how to help with that. Could you please clarify?"
        prompt = (
            "You are a helpful assistant for AutoStream, a SaaS video editing product.\n"
            "The user said something outside the known pricing, policy, and lead-capture flow.\n"
            "Respond briefly, politely, and ask a clarifying question.\n\n"
            f"User message: {state['input']}\n"
        )
        state["response"] = generate_grounded_response(prompt, fallback_text)

    return state


def router(state):
    query = state["input"].lower()
    intent = state.get("intent")

    if state.get("lead_stage"):
        return "lead_collect"

    if any(word in query for word in ["policy", "policies", "refund", "support"]):
        return "policy"

    if intent == "high_intent":
        return "lead"

    if any(word in query for word in ["price", "pricing", "plan", "cost"]):
        return "pricing"

    if intent == "greeting":
        return "greeting"

    return "fallback"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("intent", intent_node)
    graph.add_node("policy", policy_node)
    graph.add_node("pricing", pricing_node)
    graph.add_node("lead", lead_node)
    graph.add_node("lead_collect", lead_collect_node)
    graph.add_node("greeting", greeting_node)
    graph.add_node("fallback", fallback_node)

    graph.set_entry_point("intent")

    graph.add_conditional_edges(
        "intent",
        router,
        {
            "policy": "policy",
            "pricing": "pricing",
            "lead": "lead",
            "lead_collect": "lead_collect",
            "greeting": "greeting",
            "fallback": "fallback",
        },
    )

    graph.add_edge("policy", END)
    graph.add_edge("pricing", END)
    graph.add_edge("lead", END)
    graph.add_edge("lead_collect", END)
    graph.add_edge("greeting", END)
    graph.add_edge("fallback", END)

    return graph.compile()
