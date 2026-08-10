"""
Builds a semantic embedding index over all product manuals in manuals/.

Uses nomic-embed-text via Ollama (free, fully local, no API key needed).
Semantic embeddings handle paraphrased/conversational questions far better
than TF-IDF — "Is it safe to fly in the rain?" finds "Flight Environment
Requirements" even with no shared vocabulary.

Run once (or whenever manuals/*.txt changes):
  ollama pull nomic-embed-text   # one-time download (~274 MB)
  python build_index.py
"""

import pickle
import re
import sys
from pathlib import Path

import numpy as np
import requests

BASE_DIR = Path(__file__).parent
MANUALS_DIR = BASE_DIR / "manuals"
INDEX_PATH = BASE_DIR / "index.pkl"

EMBED_MODEL = "nomic-embed-text"
OLLAMA_URL = "http://localhost:11434"


def embed_text(text: str) -> list[float]:
    resp = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def parse_manual(text: str):
    chunks = []
    current_source = "HOVERAir Manual"
    current_title = None
    current_chunk_source = current_source
    current_lines = []

    def flush():
        if current_title and current_lines:
            content = "\n".join(current_lines).strip()
            if content:
                chunks.append({
                    "source": current_chunk_source,
                    "title": current_title,
                    "content": content,
                })

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        source_match = re.match(r"^SOURCE:\s*(.+)$", line)
        section_match = re.match(r"^SECTION:\s*(.+)$", line)

        if source_match:
            current_source = source_match.group(1).strip()
            continue
        if section_match:
            flush()
            current_title = section_match.group(1).strip()
            current_chunk_source = current_source
            current_lines = []
            continue
        if line.strip() == "---":
            continue
        current_lines.append(line)

    flush()
    return chunks


def build():
    try:
        requests.get(OLLAMA_URL, timeout=3)
    except Exception:
        print("ERROR: Ollama is not running. Start Ollama and try again.", file=sys.stderr)
        sys.exit(1)

    manual_files = sorted(MANUALS_DIR.glob("*.txt"))
    if not manual_files:
        print(f"No .txt manual files found in {MANUALS_DIR}", file=sys.stderr)
        sys.exit(1)

    all_chunks = []
    for path in manual_files:
        text = path.read_text(encoding="utf-8")
        chunks = parse_manual(text)
        if not chunks:
            print(f"Warning: no SECTION: chunks in {path.name} — skipping.", file=sys.stderr)
            continue
        print(f"Parsed {len(chunks)} sections from {path.name}")
        all_chunks.extend(chunks)

    if not all_chunks:
        print("No chunks parsed. Check SECTION: markers.", file=sys.stderr)
        sys.exit(1)

    print(f"\nEmbedding {len(all_chunks)} sections with {EMBED_MODEL}...")
    embeddings = []
    for i, c in enumerate(all_chunks):
        text = f"{c['title']}. {c['content']}"
        emb = embed_text(text)
        embeddings.append(emb)
        print(f"  [{i+1}/{len(all_chunks)}] {c['title'][:65]}")

    matrix = np.array(embeddings, dtype=np.float32)

    with open(INDEX_PATH, "wb") as f:
        pickle.dump({
            "embeddings": matrix,
            "chunks": all_chunks,
            "embed_model": EMBED_MODEL,
        }, f)

    print(f"\nDone: {len(all_chunks)} sections indexed -> {INDEX_PATH}")
    for c in all_chunks:
        print(f"  - [{c['source']}] {c['title']}")


if __name__ == "__main__":
    build()
