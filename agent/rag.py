import json

def load_knowledge():
    with open("knowledge_base/data.json", "r") as f:
        return json.load(f)

def get_relevant_info(query):
    data = load_knowledge()
    query = query.lower()

    # Policy related
    policy_keywords = [
        "policy",
        "policies",
        "refund",
        "support",
        "money back",
        "contact",
        "help",
        "cancel"
    ]

    if any(word in query for word in policy_keywords):
        return data["policies"]
    
    # Pricing related
    pricing_keywords = [
        "price",
        "pricing",
        "plan",
        "plans",
        "subscription",
        "subscriptions",
        "cost",
        "video",
        "videos",
        "caption",
        "captions",
        "4k",
        "720p"
    ]

    if any(word in query for word in pricing_keywords):
        return data["pricing"]

    return None