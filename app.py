"""
Backend for the HOVERAir multi-product support chatbot.

Flow per question:
  1. FAQ cache check — return instantly if question was answered before.
  2. Semantic retrieval — embed the question with nomic-embed-text (local,
     via Ollama) and find the most relevant manual sections via cosine
     similarity. Groq has no embeddings API, so this step stays local.
  3. Inject live pricing if the question is price-related.
  4. Stream the answer token-by-token from Groq (GROQ_LLM_MODEL), with
     conversation history for follow-up question support.
  5. Cache the completed answer for future identical/paraphrased questions.

Run:
  ollama pull nomic-embed-text   # embedding model (one-time)
  # GROQ_API_KEY must be set in .env — generation, talk-mode STT, and
  # talk-mode TTS all run on Groq's cloud API, not Ollama.
  python build_index.py          # builds index.pkl from manuals/
  python app.py                  # serves on http://localhost:5001
"""

import json
import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context

import faq_cache
import price_fetcher

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")
INDEX_PATH = BASE_DIR / "index.pkl"

TOP_K = 5
EMBED_MODEL = "nomic-embed-text"
OLLAMA_URL = "http://localhost:11434"
DECLINE_MARKER = "[[NOT_IN_MANUAL]]"
CHAT_MAX_TOKENS = 500
TALK_MAX_TOKENS = 150

# Answer generation (both chat and talk mode) runs on Groq's free cloud API.
# Talk mode additionally uses Groq for STT and TTS. Only retrieval
# (embed_query, via EMBED_MODEL/OLLAMA_URL above) stays on local Ollama —
# Groq has no embeddings API — so Ollama must still be running.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_LLM_MODEL = "llama-3.1-8b-instant"
GROQ_STT_MODEL = "whisper-large-v3-turbo"
GROQ_TTS_MODEL = "canopylabs/orpheus-v1-english"
GROQ_TTS_VOICE = "troy"

app = Flask(__name__, static_folder=str(BASE_DIR / "static"))

_index_cache = None
_groq_client = None


def get_groq_client():
    global _groq_client
    if _groq_client is None:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set in .env")
        from groq import Groq
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


def load_index():
    global _index_cache
    if _index_cache is None:
        if not INDEX_PATH.exists():
            raise RuntimeError("index.pkl not found. Run `python build_index.py` first.")
        with open(INDEX_PATH, "rb") as f:
            _index_cache = pickle.load(f)
    return _index_cache


def embed_query(text: str) -> np.ndarray:
    resp = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text, "keep_alive": "30m"},
        timeout=60,
    )
    resp.raise_for_status()
    return np.array(resp.json()["embedding"], dtype=np.float32)


def cosine_similarities(query_emb: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q = query_emb / (np.linalg.norm(query_emb) + 1e-9)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
    return (matrix / norms) @ q


def retrieve(question: str, top_k: int = TOP_K):
    idx = load_index()
    embeddings = idx["embeddings"]
    chunks = idx["chunks"]

    q_emb = embed_query(question)
    sims = cosine_similarities(q_emb, embeddings)

    ranked = np.argsort(sims)[::-1]
    results = []
    for i in ranked[:top_k]:
        if sims[i] <= 0:
            continue
        results.append({**chunks[i], "score": float(sims[i])})
    return results


def build_system_prompt(sections: list, mode: str = "chat", live_prices: str = "") -> str:
    if sections:
        context_blocks = []
        for s in sections:
            context_blocks.append(f"[Source: {s['source']} | Section: {s['title']}]\n{s['content']}")
        context = "\n\n".join(context_blocks)
    else:
        context = "(No manual sections matched this question.)"

    price_block = f"\nLive pricing (fetched in real-time from hoverair.com):\n{live_prices}\n" if live_prices else ""

    decline_rule = f"""Use the manual excerpts as your knowledge base. Think about what the user
is really asking — including follow-up questions that refer to earlier messages in the conversation —
then reason across all the excerpts to give the most helpful answer possible, even if the question
is phrased differently from the manual text. Synthesize, infer, and summarize freely.
Only use what is in the excerpts — never use outside knowledge.

STRICT RULES:
1. You only support HOVERAir products: X1, X1 PRO, X1 PROMAX, and AQUA. Never answer questions
   about any other brand or product (DJI, GoPro, Autel, Sony, or any other company).
2. Never mention, link to, or recommend any other company, brand, or website.
3. If the question is about another brand or is completely outside HOVERAir products,
   begin your reply with the exact marker "{DECLINE_MARKER}" — nothing before it — then say
   in 1-2 sentences that it is outside the scope of this support bot and suggest contacting
   HOVERAir support at hoverair.com. Do not add a Source line on a decline."""

    if mode == "talk":
        return f"""You are a smart, friendly voice assistant for HOVERAir drones (X1, X1 PRO, X1 PROMAX, and AQUA).
The user is speaking to you. Use conversation history for context on follow-up questions.

{decline_rule}

Reply in plain spoken language: no markdown, no bullet lists, no headers. Keep it to 1-3 short sentences.

Manual excerpts:
---
{context}
---
{price_block}"""
    else:
        return f"""You are a smart customer support assistant for HOVERAir drones (X1, X1 PRO, X1 PROMAX, and AQUA).
Use conversation history for context on follow-up questions.

{decline_rule}

Manual excerpts:
---
{context}
---
{price_block}"""


@app.route("/")
def index_page():
    return send_from_directory(str(BASE_DIR / "static"), "index.html")


@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "audio file is required"}), 400
    audio_file = request.files["audio"]
    try:
        client = get_groq_client()
        result = client.audio.transcriptions.create(
            file=(audio_file.filename or "recording.webm", audio_file.read()),
            model=GROQ_STT_MODEL,
            response_format="text",
        )
        text = result if isinstance(result, str) else getattr(result, "text", "")
        return jsonify({"text": text.strip()})
    except Exception as e:
        app.logger.error("Groq transcription failed: %s", e)
        return jsonify({"error": f"Transcription failed: {e}"}), 500


@app.route("/api/tts", methods=["POST"])
def tts():
    data = request.get_json(force=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400
    try:
        client = get_groq_client()
        speech = client.audio.speech.create(
            model=GROQ_TTS_MODEL,
            voice=GROQ_TTS_VOICE,
            input=text,
            response_format="wav",
        )
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        speech.write_to_file(tmp_path)
        with open(tmp_path, "rb") as f:
            audio_bytes = f.read()
        os.remove(tmp_path)
        return Response(audio_bytes, mimetype="audio/wav")
    except Exception as e:
        app.logger.error("Groq TTS failed: %s", e)
        return jsonify({"error": f"Speech synthesis failed: {e}"}), 500


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True) or {}
    question = (data.get("question") or "").strip()
    mode = data.get("mode") if data.get("mode") in ("chat", "talk") else "chat"
    history = data.get("history") or []

    if not question:
        return jsonify({"error": "question is required"}), 400

    # --- FAQ cache check ---
    # Only for standalone questions: an answer generated with conversation
    # history can bake in context ("you're asking about X again") that
    # isn't valid to replay for an unrelated future question that happens
    # to match on wording.
    cached = faq_cache.lookup(question, mode) if not history else None
    if cached is not None:
        app.logger.info("FAQ cache HIT (mode=%s): %r", mode, question)
        def stream_cached():
            yield f"data: {json.dumps({'token': cached['answer']})}\n\n"
            yield f"data: {json.dumps({'done': True, 'sources': cached['sources'], 'answer': cached['answer']})}\n\n"
        return Response(stream_with_context(stream_cached()), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # --- Retrieval ---
    try:
        sections = retrieve(question)
    except requests.exceptions.Timeout:
        def stream_error():
            yield f"data: {json.dumps({'error': 'The embedding model is taking too long to respond (it may still be loading into memory). Please try again in a moment.'})}\n\n"
        return Response(stream_with_context(stream_error()), mimetype="text/event-stream")
    except requests.exceptions.ConnectionError:
        def stream_error():
            yield f"data: {json.dumps({'error': 'Could not reach Ollama. Make sure it is running (ollama serve).'})}\n\n"
        return Response(stream_with_context(stream_error()), mimetype="text/event-stream")
    except Exception as e:
        err_msg = str(e)
        def stream_error():
            yield f"data: {json.dumps({'error': f'Embedding failed: {err_msg}'})}\n\n"
        return Response(stream_with_context(stream_error()), mimetype="text/event-stream")

    # --- Live pricing injection ---
    live_prices = ""
    if price_fetcher.is_price_question(question):
        live_prices = price_fetcher.get_live_prices()
        if live_prices:
            app.logger.info("Live prices injected")

    system_prompt = build_system_prompt(sections, mode, live_prices)
    max_tokens = TALK_MAX_TOKENS if mode == "talk" else CHAT_MAX_TOKENS

    # --- Build message array with conversation history ---
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-6:]:
        if isinstance(msg, dict) and msg.get("role") in ("user", "assistant") and msg.get("content"):
            messages.append({"role": msg["role"], "content": str(msg["content"])})
    messages.append({"role": "user", "content": question})

    # --- Stream response ---
    def generate():
        full_answer = [""]

        try:
            app.logger.info("Streaming from Groq model=%s", GROQ_LLM_MODEL)
            client = get_groq_client()
            stream = client.chat.completions.create(
                model=GROQ_LLM_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
                stream=True,
            )
            for chunk in stream:
                token = chunk.choices[0].delta.content
                if token:
                    full_answer[0] += token
                    yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as e:
            app.logger.error("Groq streaming failed: %s", e)
            yield f"data: {json.dumps({'error': f'Groq error: {e}'})}\n\n"
            return

        answer_text = full_answer[0].strip()
        if answer_text.startswith(DECLINE_MARKER):
            answer_text = answer_text[len(DECLINE_MARKER):].strip()
            sources_out = []
        else:
            sources_out = [{"title": s["title"], "source": s["source"]} for s in sections]

        if not history:
            faq_cache.store(question, mode, answer_text, sources_out)
            app.logger.info("FAQ cache MISS (mode=%s): %r — streamed and cached", mode, question)
        else:
            app.logger.info("FAQ cache SKIP (mode=%s, has history): %r — not cached", mode, question)
        yield f"data: {json.dumps({'done': True, 'sources': sources_out, 'answer': answer_text})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    load_index()
    app.run(host="0.0.0.0", port=5001, debug=True, threaded=True)
