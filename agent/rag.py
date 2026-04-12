import json

def load_knowledge():
    with open("knowledge_base/data.json", "r") as f:
        return json.load(f)

def get_relevant_info(query):
    data = load_knowledge()
    query = query.lower()

    # Policy related
    if any(word in query for word in ["policy", "policies", "refund", "support"]):
        return data["policies"]
    
    # Pricing related
    if any(word in query for word in ["price", "pricing", "plan", "cost"]):
        return data["pricing"]

    return "Sorry, I couldn't find relevant information."