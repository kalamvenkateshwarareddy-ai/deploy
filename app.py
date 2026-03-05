import os
import re
import uuid
import json
from typing import Any, Dict, List

import boto3
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

load_dotenv()


# ─────────────────────────────────────────────────────────
# AWS Bedrock — generates embeddings via Titan Embed Text v2
# No model download. Just an API call.
# ─────────────────────────────────────────────────────────

def get_embedding(text: str, client, model_id: str) -> List[float]:
    body = json.dumps({"inputText": text, "dimensions": 512, "normalize": True})
    response = client.invoke_model(
        modelId=model_id,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    return json.loads(response["body"].read())["embedding"]


# ─────────────────────────────────────────────────────────
# OpenSearch client
# ─────────────────────────────────────────────────────────

def get_opensearch_client(host: str, region: str) -> OpenSearch:
    credentials = boto3.Session().get_credentials()
    awsauth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        region,
        "es",
        session_token=credentials.token,
    )
    return OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
    )


def ensure_index(os_client: OpenSearch, index_name: str, dims: int = 512):
    """Create the OpenSearch index with knn vector mapping if it doesn't exist."""
    if os_client.indices.exists(index=index_name):
        return
    body = {
        "settings": {"index": {"knn": True}},
        "mappings": {
            "properties": {
                "token":     {"type": "keyword"},
                "source":    {"type": "keyword"},
                "chunk_num": {"type": "integer"},
                "text":      {"type": "text"},
                "embedding": {
                    "type":      "knn_vector",
                    "dimension": dims,
                    "method": {
                        "name":       "hnsw",
                        "space_type": "cosinesimil",
                        "engine":     "nmslib",
                    },
                },
            }
        },
    }
    os_client.indices.create(index=index_name, body=body)


# ─────────────────────────────────────────────────────────
# Text chunking (unchanged from original)
# ─────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be > overlap")
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = end - overlap
    return chunks


# ─────────────────────────────────────────────────────────
# Index a file into OpenSearch
# ─────────────────────────────────────────────────────────

def index_single_file(
    file_path: str,
    token: str,
    os_client: OpenSearch,
    bedrock_client,
    bedrock_model: str,
    index_name: str,
    chunk_size: int,
    overlap: int,
) -> int:
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().strip()
    if not text:
        raise ValueError("Text file is empty.")

    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    abs_path = os.path.abspath(file_path)

    # Delete any existing chunks for this token + file so re-upload is clean
    os_client.delete_by_query(
        index=index_name,
        body={"query": {"bool": {"must": [
            {"term": {"token": token}},
            {"term": {"source": abs_path}},
        ]}}},
        refresh=True,
        ignore_unavailable=True,
    )

    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk, bedrock_client, bedrock_model)
        doc = {
            "token":     token,
            "source":    abs_path,
            "chunk_num": i,
            "text":      chunk,
            "embedding": embedding,
        }
        doc_id = f"{token}::{abs_path}::chunk_{i}"
        os_client.index(index=index_name, id=doc_id, body=doc, refresh=False)

    os_client.indices.refresh(index=index_name)
    return len(chunks)


# ─────────────────────────────────────────────────────────
# Retrieve top-k chunks from OpenSearch
# ─────────────────────────────────────────────────────────

def retrieve(
    question: str,
    token: str,
    os_client: OpenSearch,
    bedrock_client,
    bedrock_model: str,
    index_name: str,
    k: int,
) -> List[Dict]:
    query_embedding = get_embedding(question, bedrock_client, bedrock_model)

    query_body = {
        "size": k,
        "query": {
            "bool": {
                "must": [{"term": {"token": token}}],
                "should": [{
                    "knn": {
                        "embedding": {
                            "vector": query_embedding,
                            "k": k,
                        }
                    }
                }],
            }
        },
    }

    response = os_client.search(index=index_name, body=query_body)
    hits = response["hits"]["hits"]

    return [
        {
            "document": h["_source"]["text"],
            "metadata": {
                "source":    h["_source"]["source"],
                "chunk":     h["_source"]["chunk_num"],
                "token":     h["_source"]["token"],
            },
            "score": h["_score"],
        }
        for h in hits
    ]


# ─────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────

def _base_config() -> Dict[str, Any]:
    return {
        # Bedrock
        "bedrock_region": os.getenv("BEDROCK_REGION", "us-east-1"),
        "bedrock_model":  os.getenv("BEDROCK_MODEL",  "amazon.titan-embed-text-v2:0"),
        # OpenSearch
        "opensearch_host":  os.getenv("OPENSEARCH_HOST"),   # set this after creating the domain
        "opensearch_region": os.getenv("OPENSEARCH_REGION", "us-east-1"),
        "opensearch_index":  os.getenv("OPENSEARCH_INDEX",  "rag_chunks"),
        # Chunking
        "chunk_size": int(os.getenv("CHUNK_SIZE", "900")),
        "overlap":    int(os.getenv("OVERLAP",    "150")),
        "k":          int(os.getenv("K",          "4")),
    }


def _step(name, status, detail=""):
    return {"step": name, "status": status, "detail": detail}


# ─────────────────────────────────────────────────────────
# Flask App
# ─────────────────────────────────────────────────────────

def create_app() -> Flask:
    app = Flask(__name__)
    config = _base_config()

    if not config["opensearch_host"]:
        raise RuntimeError("OPENSEARCH_HOST env variable is not set. See DEPLOY.md.")

    # Create shared clients once at startup
    bedrock_client = boto3.client("bedrock-runtime", region_name=config["bedrock_region"])
    os_client      = get_opensearch_client(config["opensearch_host"], config["opensearch_region"])

    # Create the index (does nothing if it already exists)
    ensure_index(os_client, config["opensearch_index"])

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/upload")
    def upload_file():
        body = request.get_json()
        if not body:
            return jsonify({"error": "No JSON body received"}), 400

        file_name = body.get("file_name")
        token     = body.get("token") or uuid.uuid4().hex
        steps: List[Dict] = []

        if not file_name:
            steps.append(_step("validate_request", "failed", "file_name is required"))
            return jsonify({"status": "failed", "steps": steps}), 400

        try:
            steps.append(_step("validate_request", "completed", "request validated"))
            n_chunks = index_single_file(
                file_path=file_name,
                token=token,
                os_client=os_client,
                bedrock_client=bedrock_client,
                bedrock_model=config["bedrock_model"],
                index_name=config["opensearch_index"],
                chunk_size=config["chunk_size"],
                overlap=config["overlap"],
            )
            steps.append(_step("index_to_opensearch", "completed", f"indexed {n_chunks} chunks"))
        except FileNotFoundError:
            steps.append(_step("read_file", "failed", f"file not found: {file_name}"))
            return jsonify({"status": "failed", "token": token, "steps": steps}), 404
        except ValueError as exc:
            steps.append(_step("process", "failed", str(exc)))
            return jsonify({"status": "failed", "token": token, "steps": steps}), 400
        except Exception as exc:
            steps.append(_step("index_to_opensearch", "failed", str(exc)))
            return jsonify({"status": "failed", "token": token, "steps": steps}), 500

        return jsonify({"status": "completed", "token": token, "steps": steps})

    @app.post("/query")
    def query():
        body     = request.get_json(silent=True) or {}
        question = body.get("question")
        token    = body.get("token")
        k        = int(body.get("k", config["k"]))

        if not question or not token:
            return jsonify({"status": "failed", "detail": "question and token are required"}), 400

        try:
            hits = retrieve(
                question=question,
                token=token,
                os_client=os_client,
                bedrock_client=bedrock_client,
                bedrock_model=config["bedrock_model"],
                index_name=config["opensearch_index"],
                k=k,
            )
        except Exception as exc:
            return jsonify({"status": "failed", "detail": str(exc)}), 500

        return jsonify({"status": "completed", "token": token, "top_k": hits})

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
