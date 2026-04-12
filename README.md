# AutoStream Social-to-Lead Agent

This project is a LangGraph-based conversational AI agent for a fictional SaaS product called `AutoStream`, an automated video editing platform for content creators. The agent answers pricing and policy questions from a local knowledge base, identifies high-intent users, collects lead details across multiple turns, and triggers a mock lead capture tool only after all required details are gathered.

## Features

- Intent detection for greetings, product inquiries, and high-intent lead signals
- Local RAG using `knowledge_base/data.json`
- LangGraph workflow orchestration
- Multi-turn memory for lead qualification
- Safe tool execution using `mock_lead_capture(name, email, platform)`
- Grounded response generation using `Gemini 1.5 Flash` with deterministic fallbacks

## Project Structure

```text
social-to-lead-agent/
|-- app.py
|-- README.md
|-- requirements.txt
|-- knowledge_base/
|   `-- data.json
`-- agent/
    |-- graph.py
    |-- intent.py
    |-- rag.py
    |-- tools.py
    `-- workflow.py
```

## How To Run Locally

1. Create and activate a virtual environment.

```powershell
python -m venv venv
.\venv\Scripts\activate
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
```

3. Create a `.env` file in the project root.

```env
GEMINI_API_KEY=your_api_key_here
```

4. Run the application.

```powershell
python app.py
```

5. Try a sample conversation.

```text
You: Hi, tell me about your pricing.
You: That sounds good, I want to try the Pro plan for my YouTube channel.
You: Anish
You: anish@example.com
You: YouTube
```

## Architecture Explanation

I chose `LangGraph` because this assignment needs more than a single chatbot response. The agent has to classify user intent, retrieve grounded product information, preserve memory across multiple turns, and trigger a tool only when the conversation reaches the correct stage. LangGraph is a strong fit for this because it provides a clean graph-based workflow where each node has a single responsibility and state is explicitly passed between steps.

The workflow starts with an intent node that classifies the user message. A router then decides whether the message should go to pricing, policy, greeting, lead qualification, or fallback handling. Pricing and policy answers are grounded using the local JSON knowledge base in `knowledge_base/data.json`. For natural-language responses, the agent uses `Gemini 1.5 Flash`, while deterministic fallback text ensures the app still behaves correctly if the model is unavailable.

State is managed through LangGraph-compatible typed state plus application-level turn persistence in `app.py`. Each new user message is injected as fresh `input`, while long-lived fields such as `lead_stage`, `name`, `email`, and `platform` are preserved across turns. This design prevents stale input from overwriting the current request and allows the agent to complete a 5-6 turn lead capture flow safely.

## State Management

The project uses two layers of state handling:

- `LangGraph state schema` in `agent/graph.py` defines the fields that move through the graph
- `Turn persistence` in `app.py` stores only durable conversation fields between turns

Transient values such as `input`, `intent`, and `response` are not carried into the next turn. Persistent values such as `lead_stage`, `name`, `email`, and `platform` are retained until lead capture is completed. This avoids `KeyError: 'input'` problems and prevents previous state from overwriting new user input.

## RAG Design

The RAG layer is intentionally local and lightweight for this assignment:

- Knowledge is stored in `knowledge_base/data.json`
- `agent/rag.py` loads the local file and returns pricing or policy information based on the query
- `agent/graph.py` uses that retrieved information to build grounded responses

This keeps the answers deterministic, easy to inspect, and aligned with the assignment requirement to use a local JSON or Markdown knowledge base.

## Tool Execution Logic

The lead capture tool is implemented in `agent/tools.py`.

The tool is called only after the agent has collected:

- Name
- Email
- Creator platform

The graph stores the current collection stage in `lead_stage`. This ensures the tool is never triggered prematurely.

## Example Conversation Flow

1. User asks about pricing.
2. Agent retrieves pricing from the local knowledge base and responds.
3. User expresses buying intent.
4. Agent switches to lead qualification mode.
5. Agent collects name, email, and creator platform across multiple turns.
6. Agent calls `mock_lead_capture(...)` and confirms success.

## WhatsApp Deployment Approach Using Webhooks

To integrate this agent with WhatsApp in a real deployment, I would use the WhatsApp Business Cloud API with a webhook-based backend service.

The backend would expose a webhook endpoint using a lightweight framework such as `FastAPI` or `Flask`. When WhatsApp sends an incoming message event, the webhook would parse the sender ID and message body, then use the sender's phone number as the conversation identifier. A conversation store, such as Redis or a database table, would persist the LangGraph state for each user so the agent can continue the lead flow across multiple messages.

The backend would then call the LangGraph workflow with the new incoming message and the stored state, receive the updated response, save the new state, and send the agent reply back through the WhatsApp API. If the user completes the lead collection flow, the backend could additionally forward the captured lead to a CRM or internal sales system. This architecture is scalable because message delivery is event-driven, state is keyed by user identity, and the same graph logic can serve web chat, WhatsApp, or other messaging channels.

## Dependencies

All required Python packages are listed in `requirements.txt`.

## Notes

- `.env` is ignored in `.gitignore` and should not be committed
- The core workflow uses LangGraph for orchestration and state management
- The model layer uses `Gemini 1.5 Flash`, while the local knowledge base remains the source of truth for pricing and policy data
