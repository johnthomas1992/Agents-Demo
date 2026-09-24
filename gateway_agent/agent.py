"""
Gateway / Orchestrator Agent

Routes incoming requests to specialist agents:
  - Policy / guidance      → rag_agent         (Vertex AI RAG knowledge base)
  - Escalation requests    → escalation_agent

Each specialist agent is called via HTTP, supporting both:
  - Local mode  (localhost:PORT/chat)          — set *_AGENT_URL to a localhost address
  - Deployed    (Agent Engine streamQuery URL) — set *_AGENT_URL to the aiplatform URL

Env vars (all optional — defaults assume local ports from local/run_local.py):
  RAG_AGENT_URL          http://localhost:8004/chat  OR Agent Engine streamQuery URL
  ESCALATION_AGENT_URL   http://localhost:8005/chat  OR Agent Engine streamQuery URL
"""
import logging
import os
import re

import vertexai
import vertexai.agent_engines as agent_engines
from vertexai.agent_engines import AdkApp
from google.adk.agents.llm_agent import Agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

vertexai.init(
    project=os.environ.get('GOOGLE_CLOUD_PROJECT', 'agents-demo-509203'),
    location=os.environ.get('GOOGLE_CLOUD_LOCATION', 'us-central1'),
)

RAG_AGENT_URL        = os.environ.get('RAG_AGENT_URL',        'http://localhost:8004/chat')
ESCALATION_AGENT_URL = os.environ.get('ESCALATION_AGENT_URL', 'http://localhost:8005/chat')


def _call_agent(url: str, message: str, app_name: str) -> str:
    """Calls a downstream agent.

    - localhost /chat endpoints → plain JSON POST
    - Agent Engine URLs        → Vertex AI SDK stream_query (runs full agent loop)
    """
    is_local = 'localhost' in url or '127.0.0.1' in url

    if is_local:
        import requests as http_requests
        resp = http_requests.post(
            url,
            json={"app_name": app_name, "message": message},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("reply") or "Agent returned no reply."

    # Agent Engine: raw HTTP only returns the first model turn (function_call);
    # use the SDK so the full multi-turn loop (tool call → result → text) completes.
    m = re.search(r'(projects/[^/]+/locations/[^/]+/reasoningEngines/\d+)', url)
    if not m:
        logger.error("[gateway] cannot parse resource name from url=%s", url)
        return "Invalid agent URL."
    resource_name = m.group(1)
    logger.info("[gateway] calling %s via SDK", resource_name)

    ae = agent_engines.get(resource_name)
    texts = []
    for event in ae.stream_query(user_id="gateway", message=message):
        content = event.get("content") or {}
        if content.get("role") == "user":
            continue
        for part in content.get("parts", []):
            if part.get("text"):
                texts.append(part["text"])

    result = ''.join(texts)
    logger.info("[gateway] received %d chars from %s", len(result), resource_name)
    return result or "Agent returned no text response."


def call_rag_agent(message: str) -> dict:
    """Queries the RAG policy and guidance agent for policy, fee, product and how-to questions.

    Args:
        message: The user's policy or product question.
    """
    logger.info("[gateway] → rag_agent: %s", message)
    try:
        reply = _call_agent(RAG_AGENT_URL, message, "rag_agent")
        return {"status": "success", "reply": reply}
    except Exception as e:
        logger.error("[gateway] rag_agent failed: %s", e)
        return {"status": "error", "error": str(e)}


def call_escalation_agent(message: str) -> dict:
    """Escalates an unresolved issue to the human queue via the escalation agent.

    Args:
        message: A description of the issue and what has already been tried.
    """
    logger.info("[gateway] → escalation_agent: %s", message)
    try:
        reply = _call_agent(ESCALATION_AGENT_URL, message, "escalation_agent")
        return {"status": "success", "reply": reply}
    except Exception as e:
        logger.error("[gateway] escalation_agent failed: %s", e)
        return {"status": "error", "error": str(e)}


root_agent = Agent(
    model='gemini-2.5-flash',
    name='root_agent',
    description='Enterprise gateway agent that routes requests to specialist sub-agents.',
    instruction=(
        'You are an enterprise gateway agent. Classify the user\'s intent and route '
        'to the correct specialist using these STRICT rules:\n\n'

        '1. POLICY / GUIDANCE / PRODUCT — refund policy, fees, plan details, how-tos, '
        'data retention, account suspension, product features, upgrades, downgrades:\n'
        '   → call call_rag_agent with the user\'s exact message.\n\n'

        '2. ESCALATION — user explicitly asks for a human agent, or expresses that '
        'their issue is unresolved after agent responses, or reports a critical incident:\n'
        '   → call call_escalation_agent describing the issue and prior attempts.\n\n'

        'GUARDRAILS:\n'
        '- Never answer policy or product questions from your own knowledge. '
        'Always use the appropriate tool.\n'
        '- If the topic is completely out of scope (e.g. general trivia, coding help), '
        'politely decline and explain you are an enterprise servicing assistant.\n'
        '- Pass the user\'s message to the tool verbatim — do not paraphrase.\n'
        '- After receiving a tool reply, present it clearly to the user.'
    ),
    tools=[call_rag_agent, call_escalation_agent],
)
logger.info("[gateway_agent] root_agent constructed")

adk_app = AdkApp(agent=root_agent, enable_tracing=True)
logger.info("[gateway_agent] AdkApp constructed")
