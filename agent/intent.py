def detect_intent(query):
    query = query.lower()

    # high intent (strong signals only)
    if any(phrase in query for phrase in [
        "subscribe",
        "sign up",
        "get started",
        "want to try",
        "want to join",
        "want to buy",
        "buy",
        "purchase",
        "try the pro plan",
        "try pro plan",
        "want the pro plan",
        "i want the pro plan",
        "proceed with the pro plan",
        "proceed with pro plan",
        "i would proceed",
        "i will proceed",
        "i would like to proceed",
        "i want to proceed",
        "ready to proceed",
        "ready to sign up",
        "i am ready to sign up",
        "i want to sign up",
        "i will proceed",
        "let's start",
        "i am ready"
    ]):
        return "high_intent"

    # product / info queries
    if any(word in query for word in [
        "price", "pricing", "plan", "cost",
        "feature", "details", "policy", "policies",
        "refund", "support"
    ]):
        return "product_query"

    # greeting
    if any(word in query for word in ["hi", "hello", "hey"]):
        return "greeting"

    return "unknown"
