import logging
import os
import yaml
import google.auth
from google.cloud import bigquery

logger = logging.getLogger(__name__)


def _load_query(name: str) -> str:
    yaml_path = os.path.join(os.path.dirname(__file__), '..', 'queries.yaml')
    logger.info("[tools] Loading query '%s' from %s", name, os.path.abspath(yaml_path))
    with open(yaml_path) as f:
        queries = yaml.safe_load(f)
    sql = queries['queries'][name]
    logger.info("[tools] Query loaded: %s", sql.strip())
    return sql


def fetch_potassium_labs(days: int = 0) -> dict:
    """Fetch entries from the potassium_labs BigQuery table.

    Args:
        days: Number of past days to fetch (e.g. 30 for last 30 days).
              Pass 0 or omit to fetch all rows.
    """
    logger.info("[tools] fetch_potassium_labs called (days=%s)", days)

    logger.info("[tools] Initialising BigQuery client (ADC)")
    try:
        credentials, _ = google.auth.default()
        client = bigquery.Client(credentials=credentials, project='agents-demo-509203')
        logger.info("[tools] BigQuery client ready (project=%s)", client.project)
    except Exception as e:
        logger.error("[tools] Failed to initialise BigQuery client: %s", e, exc_info=True)
        raise

    try:
        if days and days > 0:
            template = _load_query('fetch_potassium_labs_days')
            query = template.format(days=int(days))
        else:
            query = _load_query('fetch_potassium_labs_all')
    except Exception as e:
        logger.error("[tools] Failed to load query: %s", e, exc_info=True)
        raise

    logger.info("[tools] Submitting BigQuery job")
    try:
        job = client.query(query)
        logger.info("[tools] Job submitted (job_id=%s), waiting for results", job.job_id)
        results = job.result()
        logger.info("[tools] Job complete")
    except Exception as e:
        logger.error("[tools] BigQuery job failed: %s", e, exc_info=True)
        raise

    def _serialize(v):
        if hasattr(v, 'isoformat'):
            return v.isoformat()
        return v

    rows = [{k: _serialize(v) for k, v in dict(row).items()} for row in results]
    logger.info("[tools] Returning %d rows", len(rows))
    return {"status": "success", "row_count": len(rows), "data": rows}
