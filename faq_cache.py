"""
FAQ cache: reuses a past answer when a new question is a close match for
one already asked (by any user, chat or talk), so near-duplicate phrasing
doesn't cost another Ollama generation call each time.

Matching uses TF-IDF + cosine-similarity over past question text — a
different (cheaper, dependency-light) technique than app.py's retrieve(),
which uses nomic-embed-text semantic embeddings for manual sections. Chat
and talk are cached separately since their answer style/length differ (see
build_system_prompt()).
"""

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).parent
CACHE_PATH = BASE_DIR / "faq_cache.json"

# Cosine similarity (0-1) a new question must reach against a past question,
# within the same mode, to be served from cache instead of calling Claude.
# Short questions have very few content words after stopword removal, so a
# couple of shared words (e.g. "fly", "drone") can push cosine similarity
# past a low threshold even for unrelated questions — "How do I fly the
# drone?" vs "How long can I fly this drone on a full charge?" scored 0.55.
# Kept high so only near-identical rewordings hit the cache; anything looser
# risks serving a wrong cached answer with full confidence.
FAQ_MATCH_THRESHOLD = 0.82

_cache = None


def _load():
    global _cache
    if _cache is None:
        if CACHE_PATH.exists():
            with open(CACHE_PATH, "r") as f:
                _cache = json.load(f)
        else:
            _cache = []
    return _cache


def _save():
    with open(CACHE_PATH, "w") as f:
        json.dump(_cache, f, indent=2)


def lookup(question: str, mode: str):
    entries = [e for e in _load() if e["mode"] == mode]
    if not entries:
        return None

    corpus = [e["question"] for e in entries] + [question]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(corpus)

    sims = cosine_similarity(matrix[-1], matrix[:-1])[0]
    best_i = max(range(len(entries)), key=lambda i: sims[i])
    if sims[best_i] >= FAQ_MATCH_THRESHOLD:
        return entries[best_i]
    return None


def store(question: str, mode: str, answer: str, sources: list):
    cache = _load()
    cache.append(
        {"question": question, "mode": mode, "answer": answer, "sources": sources}
    )
    _save()
