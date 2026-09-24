import json
import logging
import os

import vertexai
from google.cloud import aiplatform

logger = logging.getLogger(__name__)

_PROJECT          = os.environ.get("GOOGLE_CLOUD_PROJECT", "agents-demo-509203")
_LOCATION         = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
_ENDPOINT_RESOURCE = os.environ.get(
    "RAG_ENDPOINT_RESOURCE",
    "projects/215815760614/locations/us-central1/indexEndpoints/7650566832809574400",
)
_DEPLOYED_INDEX_ID = os.environ.get("RAG_DEPLOYED_INDEX_ID", "enterprise_docs")
_CHUNK_STORE_PATH  = os.environ.get(
    "RAG_CHUNK_STORE_PATH",
    # Bundled alongside agent for deployed mode; fall back to repo path locally
    os.path.join(os.path.dirname(__file__), "..", "chunk_store.json"),
)

# Load chunk store once at startup
def _load_chunk_store() -> dict:
    path = os.path.abspath(_CHUNK_STORE_PATH)
    if not os.path.exists(path):
        logger.warning("[rag_tool] chunk_store.json not found at %s — run rag_setup/index_docs.py", path)
        return {}
    with open(path) as f:
        return json.load(f)

_CHUNK_STORE: dict = _load_chunk_store()

vertexai.init(project=_PROJECT, location=_LOCATION)
aiplatform.init(project=_PROJECT, location=_LOCATION)


def _embed_query(text: str) -> list[float]:
    from vertexai.language_models import TextEmbeddingModel
    model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    return model.get_embeddings([text])[0].values


_SCORE_THRESHOLD = float(os.environ.get("RAG_SCORE_THRESHOLD", "0.4"))


def retrieve_policy_docs(query: str, top_k: int = 5) -> dict:
    """Retrieves relevant policy, product and guidance documents for a query.

    Args:
        query: The question or topic to search the knowledge base for.
        top_k: Number of document chunks to retrieve (1-10, default 5).
    """
    logger.info("[rag_tool] Retrieving docs: query=%r top_k=%d threshold=%.2f",
                query, top_k, _SCORE_THRESHOLD)

    if not _CHUNK_STORE:
        return {"status": "error", "error": "Chunk store not loaded — run rag_setup/index_docs.py first"}

    try:
        # Embed query
        query_embedding = _embed_query(query)

        # Query Vector Search endpoint — fetch top_k, then filter by score
        endpoint = aiplatform.MatchingEngineIndexEndpoint(_ENDPOINT_RESOURCE)
        results = endpoint.find_neighbors(
            deployed_index_id=_DEPLOYED_INDEX_ID,
            queries=[query_embedding],
            num_neighbors=min(int(top_k), 10),
        )

        # Look up chunks and apply similarity threshold
        chunks = []
        for neighbor in results[0]:
            score = round(float(neighbor.distance), 4)
            if score < _SCORE_THRESHOLD:
                logger.info("[rag_tool] Dropping chunk %s (score=%.4f < threshold=%.2f)",
                            neighbor.id, score, _SCORE_THRESHOLD)
                continue
            entry = _CHUNK_STORE.get(neighbor.id)
            if entry:
                chunks.append({
                    "text": entry["text"],
                    "source": entry["source"],
                    "score": score,
                })

        logger.info("[rag_tool] %d/%d chunks passed threshold", len(chunks), len(results[0]))

        if not chunks:
            return {"status": "no_results", "message": "No sufficiently relevant documents found for this query."}

        return {"status": "success", "count": len(chunks), "chunks": chunks}

    except Exception as e:
        logger.error("[rag_tool] Retrieval failed: %s", e, exc_info=True)
        return {"status": "error", "error": str(e)}
