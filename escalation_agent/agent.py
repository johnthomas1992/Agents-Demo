import json
import logging
import os
from datetime import datetime, timezone

import requests as http_requests
from vertexai.agent_engines import AdkApp
from google.adk.agents.llm_agent import Agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SENDGRID_API_KEY   = os.environ.get("SENDGRID_API_KEY", "")
ESCALATION_EMAIL_TO   = os.environ.get("ESCALATION_EMAIL_TO", "john.jtn@gmail.com")
ESCALATION_EMAIL_FROM = os.environ.get("ESCALATION_EMAIL_FROM", "")


def package_escalation(
    issue_summary: str,
    severity: str = "normal",
    contact_preference: str = "email",
) -> dict:
    """Packages the conversation context into a structured escalation case for the human queue.

    Args:
        issue_summary: A concise summary of the issue, including what has already been tried.
        severity: One of 'low', 'normal', 'high', 'critical'. Default is 'normal'.
        contact_preference: How the customer prefers to be contacted: 'email', 'phone', or 'chat'.
    """
    valid_severities = {"low", "normal", "high", "critical"}
    if severity not in valid_severities:
        severity = "normal"

    case = {
        "status": "escalated",
        "case": {
            "summary": issue_summary,
            "severity": severity,
            "contact_preference": contact_preference,
            "channel": "chat",
            "escalated_at": datetime.now(timezone.utc).isoformat(),
            "estimated_wait_minutes": {"low": 60, "normal": 30, "high": 10, "critical": 5}[severity],
        },
        "message": (
            f"Your case has been escalated to a human agent (severity: {severity}). "
            f"Estimated wait: {{'low': '~60 min', 'normal': '~30 min', 'high': '~10 min', 'critical': '~5 min'}}[severity]. "
            "You will be contacted via your preferred channel."
        ),
    }
    logger.info("[escalation] Case packaged: severity=%s", severity)
    return case


def send_escalation_email(subject: str, body: str) -> dict:
    """Sends an escalation alert email to the support team via SendGrid.

    Call this AFTER package_escalation to notify the human queue.

    Args:
        subject: Email subject line, e.g. 'Escalation [HIGH]: billing issue unresolved'.
        body: Full email body summarising the case details.
    """
    if not SENDGRID_API_KEY:
        logger.warning("[escalation] SENDGRID_API_KEY not set — skipping email")
        return {"status": "skipped", "reason": "SENDGRID_API_KEY not configured"}
    if not ESCALATION_EMAIL_FROM:
        logger.warning("[escalation] ESCALATION_EMAIL_FROM not set — skipping email")
        return {"status": "skipped", "reason": "ESCALATION_EMAIL_FROM not configured"}

    payload = {
        "personalizations": [{"to": [{"email": ESCALATION_EMAIL_TO}]}],
        "from": {"email": ESCALATION_EMAIL_FROM},
        "subject": subject,
        "content": [{"type": "text/plain", "value": body}],
    }
    try:
        resp = http_requests.post(
            "https://api.sendgrid.com/v3/mail/send",
            headers={
                "Authorization": f"Bearer {SENDGRID_API_KEY}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
            timeout=15,
        )
        if resp.status_code == 202:
            logger.info("[escalation] email sent to %s", ESCALATION_EMAIL_TO)
            return {"status": "sent", "to": ESCALATION_EMAIL_TO}
        else:
            logger.error("[escalation] SendGrid error %s: %s", resp.status_code, resp.text)
            return {"status": "error", "code": resp.status_code, "detail": resp.text}
    except Exception as e:
        logger.error("[escalation] email failed: %s", e)
        return {"status": "error", "detail": str(e)}


root_agent = Agent(
    model='gemini-2.5-flash-lite',
    name='root_agent',
    description='Escalation agent that packages context, routes unresolved issues to the human queue, and emails the support team.',
    instruction=(
        'You are an escalation coordinator. Your job is to:\n'
        '1. Summarise the customer\'s issue clearly and concisely.\n'
        '2. Determine the appropriate severity:\n'
        '   - critical: data loss, security breach, complete service outage\n'
        '   - high: significant feature unavailability, billing errors\n'
        '   - normal: general support questions not resolved by automated agents\n'
        '   - low: feedback, non-urgent requests\n'
        '3. Call package_escalation with the summary, severity, and contact preference.\n'
        '4. Call send_escalation_email with a clear subject and body summarising the case.\n'
        '   Subject format: "Escalation [SEVERITY]: <brief issue title>"\n'
        '   Body should include: issue summary, severity, timestamp, and contact preference.\n'
        '5. Reassure the customer that a human agent will follow up.'
    ),
    tools=[package_escalation, send_escalation_email],
)
logger.info("[escalation_agent] root_agent constructed")

adk_app = AdkApp(agent=root_agent, enable_tracing=True)
logger.info("[escalation_agent] AdkApp constructed")
