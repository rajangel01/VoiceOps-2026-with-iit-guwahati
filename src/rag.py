
import re
import json
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
KB_DIR = ROOT / "kb"
INDEX_FILE = ROOT / "data" / "rag_index.json"

EMBED_MODEL = "nomic-embed-text"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"


def split_into_chunks(text, max_chars=1800):
    """Split Markdown into manageable chunks, preserving headings."""
    chunks = []
    current = []
    current_length = 0

    for line in text.splitlines():
        line = line.rstrip()
        if not line:
            continue

        # Start a new chunk at a Markdown heading when possible.
        if line.startswith("#") and current:
            chunks.append("\n".join(current))
            current = []
            current_length = 0

        # Split very long sections at paragraph/line boundaries.
        if current and current_length + len(line) > max_chars:
            chunks.append("\n".join(current))
            current = []
            current_length = 0

        current.append(line)
        current_length += len(line)

    if current:
        chunks.append("\n".join(current))

    return [chunk for chunk in chunks if chunk.strip()]


def get_metadata(text, filename):
    """Extract section ID, status and effective date when present."""
    section_match = re.search(
        r"\[([A-Za-z0-9_-]+)\]", text
    )
    status_match = re.search(
        r"(?im)^\s*status\s*:\s*(.+)$", text
    )
    date_match = re.search(
        r"(?im)^\s*effective_from\s*:\s*(.+)$", text
    )

    return {
        "source": filename,
        "section_id": section_match.group(1)
        if section_match else None,
        "status": status_match.group(1).strip()
        if status_match else None,
        "effective_from": date_match.group(1).strip()
        if date_match else None,
    }


def embed_texts(texts):
    """Generate embeddings locally through Ollama."""
    response = requests.post(
        OLLAMA_EMBED_URL,
        json={
            "model": EMBED_MODEL,
            "input": texts,
        },
        timeout=180,
    )
    response.raise_for_status()
    return response.json()["embeddings"]


def build_index():
    if not KB_DIR.exists():
        raise FileNotFoundError(f"Knowledge base not found: {KB_DIR}")

    documents = sorted(KB_DIR.rglob("*.md"))
    if not documents:
        raise FileNotFoundError(
            f"No Markdown files found inside {KB_DIR}"
        )

    records = []

    for path in documents:
        text = path.read_text(encoding="utf-8")
        for chunk in split_into_chunks(text):
            records.append({
                "text": chunk,
                **get_metadata(chunk, str(path.relative_to(ROOT))),
            })

    if not records:
        raise ValueError("No document chunks were created.")

    # Embed in batches to avoid oversized requests.
    for start in range(0, len(records), 16):
        batch = records[start:start + 16]
        vectors = embed_texts([item["text"] for item in batch])

        if len(vectors) != len(batch):
            raise RuntimeError("Embedding count did not match chunk count.")

        for item, vector in zip(batch, vectors):
            item["embedding"] = vector

    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    INDEX_FILE.write_text(
        json.dumps(records, ensure_ascii=False),
        encoding="utf-8",
    )

    return len(records)


def search_knowledge_base(question, top_k=4):
    if not INDEX_FILE.exists():
        raise FileNotFoundError(
            "RAG index not found. Run build_index() first."
        )

    records = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    query_vector = np.array(embed_texts([question])[0], dtype=float)

    matrix = np.array(
        [record["embedding"] for record in records],
        dtype=float,
    )

    query_norm = np.linalg.norm(query_vector)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    denominators = matrix_norms * query_norm

    scores = np.divide(
        matrix @ query_vector,
        denominators,
        out=np.zeros(len(records)),
        where=denominators != 0,
    )

    best_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in best_indices:
        record = records[int(idx)].copy()
        record.pop("embedding", None)
        record["score"] = round(float(scores[idx]), 4)
        results.append(record)

    return results
