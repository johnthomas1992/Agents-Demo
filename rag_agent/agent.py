import logging
import os

import vertexai
from vertexai.agent_engines import AdkApp
from google.adk.agents.llm_agent import Agent

from rag_agent.tools.rag_tool import retrieve_policy_docs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

vertexai.init(
    project=os.environ.get('GOOGLE_CLOUD_PROJECT', 'agents-demo-509203'),
    location=os.environ.get('GOOGLE_CLOUD_LOCATION', 'us-central1'),
)

root_agent = Agent(
    model='gemini-2.5-flash-lite',
    name='root_agent',
    description='Policy and guidance specialist that answers questions using the RAG knowledge base.',
    instruction=(
        'You are a policy and product guidance specialist.\n\n'
        'RULES:\n'
        '1. ALWAYS call retrieve_policy_docs before answering any question. '
        'Pass the user\'s question as the query.\n'
        '2. Base your answer strictly on the retrieved document chunks. '
        'Cite the source for each key fact.\n'
        '3. If the retrieved chunks do not contain enough information to answer, '
        'say clearly: "I could not find this information in our documentation."\n'
        '4. Never make up policies, fees, or product details not found in the chunks.'
    ),
    tools=[retrieve_policy_docs],
)
logger.info("[rag_agent] root_agent constructed")

adk_app = AdkApp(agent=root_agent, enable_tracing=True)
logger.info("[rag_agent] AdkApp constructed")
