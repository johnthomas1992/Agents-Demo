#!/usr/bin/env python3
"""
One-time setup: creates a Vertex AI Vector Search index + endpoint.
RAG is implemented directly on top of Vector Search (no RAG Engine).
After running this, run rag_setup/index_docs.py to embed and index the docs.

Steps performed:
  1. Create a streaming Vector Search index (768-dim, text-embedding-004)
  2. Create a public index endpoint
  3. Deploy the index to the endpoint  ← takes 20-40 min on first run
  4. Create a RAG corpus pointing at the index endpoint
  5. Upload sample documents

Usage:
    venv/bin/python3 rag_setup/create_corpus.py

Add the printed RAG_CORPUS_NAME to local/.env when done.
Auth: ADC — run `gcloud auth application-default login` first.
"""
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

PROJECT  = os.environ.get("GOOGLE_CLOUD_PROJECT", "agents-demo-509203")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
SAMPLE_DOCS_DIR = os.path.join(os.path.dirname(__file__), "sample_docs")

import vertexai
from google.cloud import aiplatform
from vertexai.preview import rag
from vertexai.preview.rag import VertexVectorSearch

vertexai.init(project=PROJECT, location=LOCATION)
aiplatform.init(project=PROJECT, location=LOCATION)


def main() -> None:
    print(f"[setup] Project: {PROJECT}  Location: {LOCATION}")

    # 1. Create Vector Search streaming index
    #    Dimensions = 768 to match text-embedding-004 (RAG Engine default)
    print("[setup] Creating Vector Search index (streaming, 768 dims) …")
    index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
        display_name="enterprise-docs-index",
        dimensions=768,
        approximate_neighbors_count=150,
        distance_measure_type="DOT_PRODUCT_DISTANCE",
        leaf_node_embedding_count=500,
        leaf_nodes_to_search_percent=7,
        index_update_method="STREAM_UPDATE",
    )
    print(f"[setup] Index created: {index.resource_name}")

    # 2. Create public index endpoint
    print("[setup] Creating index endpoint …")
    endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
        display_name="enterprise-docs-endpoint",
        public_endpoint_enabled=True,
    )
    print(f"[setup] Endpoint created: {endpoint.resource_name}")

    # 3. Deploy index to endpoint — this takes 20-40 min
    print("[setup] Deploying index to endpoint (this can take 20-40 min) …")
    endpoint = endpoint.deploy_index(
        index=index,
        deployed_index_id="enterprise_docs",
        display_name="enterprise-docs-deployed",
        min_replica_count=1,
        max_replica_count=1,
    )
    print("[setup] Index deployed.")

    # 4. Create RAG corpus backed by the Vector Search index
    print("[setup] Creating RAG corpus (Vector Search backend) …")
    corpus = rag.create_corpus(
        display_name="enterprise-docs",
        description="Policy and product documentation for the enterprise servicing agent",
        vector_db=VertexVectorSearch(
            index=index.resource_name,
            index_endpoint=endpoint.resource_name,
        ),
    )
    print(f"[setup] Corpus created: {corpus.name}")

    # 5. Upload sample documents
    docs = [
        ("policy_guide.md",  "Enterprise Policy Guide"),
        ("product_guide.md", "Product Guide"),
    ]
    for filename, display_name in docs:
        path = os.path.join(SAMPLE_DOCS_DIR, filename)
        print(f"[setup] Uploading {filename} …")
        rag_file = rag.upload_file(
            corpus_name=corpus.name,
            path=path,
            display_name=display_name,
        )
        print(f"[setup]   → {rag_file.name}")

    print("\n[setup] Done! Add this to local/.env:")
    print(f"\n    RAG_CORPUS_NAME={corpus.name}")
    print(f"\n  Index:    {index.resource_name}")
    print(f"  Endpoint: {endpoint.resource_name}\n")


if __name__ == "__main__":
    main()
