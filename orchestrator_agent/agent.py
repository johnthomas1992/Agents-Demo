import json
import logging
import os

import google.auth
import google.auth.transport.requests
import requests as http_requests
import vertexai
from google.adk.agents.llm_agent import Agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LAB_AGENT_RESOURCE = os.environ.get(
    'LAB_AGENT_RESOURCE',
    'projects/215815760614/locations/us-central1/reasoningEngines/1171537885732536320',
)

_LAB_AGENT_URL = (
    f'https://us-central1-aiplatform.googleapis.com/v1/{LAB_AGENT_RESOURCE}'
    ':streamQuery?alt=sse'
)

vertexai.init(
    project=os.environ.get('GOOGLE_CLOUD_PROJECT', 'agents-demo-509203'),
    location=os.environ.get('GOOGLE_CLOUD_LOCATION', 'us-central1'),
)


def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city."""
    return {"status": "success", "city": city, "time": "10:30 AM"}


def call_lab_agent(message: str) -> dict:
    """Delegates a lab data question to the remote lab agent deployed on Agent Engine.

    Args:
        message: The user's question about lab data, e.g. 'get me last 30 days data'.
    """
    logger.info("[orchestrator] Delegating to lab agent: %s", message)
    try:
        credentials, _ = google.auth.default()
        credentials.refresh(google.auth.transport.requests.Request())

        resp = http_requests.post(
            _LAB_AGENT_URL,
            headers={
                'Authorization': f'Bearer {credentials.token}',
                'Content-Type': 'application/json',
            },
            json={
                'class_method': 'async_stream_query',
                'input': {'message': message, 'user_id': 'orchestrator'},
            },
            timeout=120,
        )
        resp.raise_for_status()

        # Response is NDJSON (newline-delimited JSON objects, one per line)
        events = []
        for line in resp.text.splitlines():
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

        logger.info("[orchestrator] Lab agent returned %d events", len(events))

        # Each event has content.role ("model"/"user") and content.parts[].text
        for event in reversed(events):
            content = event.get('content') or {}
            role = content.get('role', '')
            parts = content.get('parts') or []
            for part in parts:
                if part.get('text') and role != 'user':
                    return {'status': 'success', 'reply': part['text']}

        return {'status': 'success', 'reply': 'Lab agent returned no text response.'}
    except Exception as e:
        logger.error("[orchestrator] Lab agent call failed: %s", e, exc_info=True)
        return {'status': 'error', 'error': str(e)}


root_agent = Agent(
    model='gemini-2.5-flash-lite',
    name='root_agent',
    description='Orchestrator that routes requests to specialised sub-agents.',
    instruction=(
        'You are an orchestrator agent with two tools: call_lab_agent and get_current_time.\n\n'
        'STRICT RULES — follow these exactly:\n'
        '1. If the user asks anything about lab data, potassium labs, lab results, '
        'patient data, or requests data for any number of days: you MUST call '
        'call_lab_agent immediately with the user\'s exact message as the argument. '
        'Do NOT attempt to answer lab questions yourself.\n'
        '2. If the user asks about the current time in a city: call get_current_time.\n'
        '3. For all other questions: tell the user you can only help with lab data '
        'queries and time lookups.\n\n'
        'You have no knowledge of lab data. call_lab_agent is the ONLY source of lab data.'
    ),
    tools=[get_current_time, call_lab_agent],
)
logger.info("[orchestrator] root_agent constructed")
