import logging

from vertexai.agent_engines import AdkApp
from google.adk.agents.llm_agent import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from remote_agent.tools.tools import fetch_potassium_labs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _ctx_info(callback_context: CallbackContext) -> str:
    """Return whatever session/invocation identifiers the context exposes."""
    attrs = {
        k: getattr(callback_context, k)
        for k in ("session_id", "invocation_id", "agent_name", "turn_id", "name")
        if hasattr(callback_context, k)
    }
    return str(attrs) if attrs else repr(callback_context)


_ctx_logged = False


def before_model_callback(callback_context: CallbackContext, llm_request: LlmRequest) -> LlmResponse | None:
    global _ctx_logged
    if not _ctx_logged:
        logger.info("[agent] CallbackContext available attrs: %s", vars(callback_context))
        _ctx_logged = True

    contents = llm_request.contents or []
    logger.info("[agent] Prompt received — ctx=%s, contents=%d", _ctx_info(callback_context), len(contents))
    last = contents[-1] if contents else None
    if last:
        logger.info("[agent] Last content role=%s parts=%s", last.role, str(last.parts)[:500])
    return None


def after_model_callback(callback_context: CallbackContext, llm_response: LlmResponse) -> LlmResponse | None:
    logger.info("[agent] Model response received — ctx=%s", _ctx_info(callback_context))
    content = llm_response.content
    if content:
        logger.info("[agent] Response parts: %s", str(content.parts)[:500])
    return None


root_agent = Agent(
    model='gemini-2.5-flash-lite',
    name='root_agent',
    description='Fetches data from the potassium_labs BigQuery table.',
    instruction='Your role is to invoke bq tool to fetch data',
    tools=[fetch_potassium_labs],
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
)

logger.info("[agent] root_agent constructed")

adk_app = AdkApp(
    agent=root_agent,
    enable_tracing=True,
)

logger.info("[agent] AdkApp constructed")