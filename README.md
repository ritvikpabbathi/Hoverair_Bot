# HOVERAir Manual Assistant (RAG demo)

A chatbot that answers questions about HOVERAir's drone lineup (X1, X1 PRO,
X1 PROMAX, AQUA) by retrieving the relevant section(s) of the official
manuals and having a locally-hosted LLM answer strictly from that text,
with citations back to the sections used. This is the prototype described
in the opportunity report: HOVERAir currently has no smart support
assistant, only static PDFs and a contact form.

Retrieval (embedding manual text to find the right section) runs locally
through **Ollama** — no API key, no data leaving your machine for that
step. Answer generation, and talk mode's speech-to-text and text-to-speech,
run on **Groq's** free-tier cloud API instead of a local model, since Groq's
hosted inference is both faster and lighter on local hardware than running
a full LLM/STT/TTS stack locally.

## How it works

1. `manuals/*.txt` holds the official Quick Start Guides and Safety
   Instructions for each product, organized into `SOURCE:` / `SECTION:`
   blocks. All manual files are indexed together, so one bot covers the
   whole product line.
2. `build_index.py` splits each manual into per-section chunks and embeds
   them with `nomic-embed-text` via Ollama, storing the result in
   `index.pkl`. Semantic embeddings (not keyword matching) mean paraphrased
   or conversational questions still find the right section even with
   little shared vocabulary — e.g. "Is it safe to fly in the rain?" finds
   "Flight Environment Requirements".
3. `app.py` is a Flask server. For each question it embeds the question via
   Ollama, retrieves the top matching manual sections, and streams an
   answer token-by-token from a Groq-hosted Llama model
   (`llama-3.1-8b-instant`) with instructions to answer only from those
   excerpts and never from outside knowledge. Follow-up questions carry
   conversation history for context. Price questions get live pricing
   pulled from HOVERAir's Shopify store (`price_fetcher.py`) injected into
   the prompt. Common/repeated first-time questions are served instantly
   from a local FAQ cache (`faq_cache.py`) instead of re-running the model.
4. `static/index.html` offers two modes, switched with a toggle at the top:
   - **Chat with the Agent** — type a question, get a written answer with
     source citation chips.
   - **Talk to the Agent** — tap the mic button and ask out loud. The
     browser records audio with `MediaRecorder` (auto-stopping after ~1.8s
     of detected silence) and uploads it to `/api/transcribe`, which sends
     it to Groq's Whisper (`whisper-large-v3-turbo`) for transcription. The
     transcribed question is submitted to `/api/chat` tagged
     `mode: "talk"`, which asks for a short, conversational, unformatted
     answer. Each sentence is spoken as soon as it finishes streaming in
     (via `/api/tts`, Groq's Orpheus TTS model) rather than waiting for the
     whole answer, with the next sentence's audio prefetched during
     playback so there's no gap between them. The full text still appears
     in the chat transcript (with source chips) as a reference.

   Both modes answer strictly from the retrieved manual excerpts and never
   state a fact that isn't in them. If a question isn't covered by the
   manuals, or is about a different brand/product, the agent doesn't just
   give a flat canned refusal — it says plainly that it's out of scope and
   points to HOVERAir support, without guessing at what the actual answer
   might be.

## Setup

```bash
cd hoverair_bot
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt

ollama pull nomic-embed-text     # embedding model (~274 MB, one-time) — only local model needed

cp .env.example .env             # then add your GROQ_API_KEY (free at console.groq.com)

./venv/bin/python build_index.py    # builds index.pkl from manuals/
./venv/bin/python app.py            # serves on http://localhost:5001
```

Ollama itself must be running (`ollama serve`, or the Ollama.app menu bar
app) before `build_index.py` or `app.py` will work — it's only used for
embeddings now, not for generation.

A `GROQ_API_KEY` is required in `.env` — get one free at
[console.groq.com](https://console.groq.com) (API Keys in the sidebar, no
billing setup needed for the free tier). The Orpheus TTS model
(`canopylabs/orpheus-v1-english`) additionally requires accepting its terms
once in the Groq console before `/api/tts` will work — visit the
[playground](https://console.groq.com/playground?model=canopylabs%2Forpheus-v1-english)
and accept when prompted.

Then open http://localhost:5001 in your browser.

## Performance

Generation now runs on Groq's cloud inference, so answers are fast
regardless of local hardware — no more 10-70s waits. The only local model
left is `nomic-embed-text` for retrieval, which is small and fast on any
machine. Groq's free tier is rate-limited rather than metered, so a burst
of rapid requests could get throttled, but normal usage is unaffected.

## Extending this to a different product

1. Add a new `manuals/<product>_manual.txt` file, using the same
   `SOURCE:` / `SECTION:` structure so chunking and citations keep working.
   Existing manual files can stay — all `.txt` files in `manuals/` are
   indexed together.
2. Re-run `python build_index.py`.
3. Update the product list in `app.py`'s `build_system_prompt()` and the
   suggested questions in `static/index.html`.
4. If the product has live pricing on the Shopify store, add its JSON
   endpoint to `PRODUCT_ENDPOINTS` in `price_fetcher.py`.

## Known limitations (worth knowing before demoing)

- This now depends on Groq's cloud API for generation, STT, and TTS — it no
  longer works fully offline, and needs a working internet connection plus
  a valid `GROQ_API_KEY`. Retrieval (embeddings) is the only step still
  fully local.
- Answers are only as good as the manual text included — it doesn't cover
  every edge case (e.g. detailed troubleshooting for specific error
  codes), since HOVERAir doesn't publish that publicly.
- This is a prototype, not a production support tool: no auth, no rate
  limiting, and the dev server binds to every network interface with
  Werkzeug's debugger active — fine for solo local use, not safe to leave
  running on a shared network.
- Talk mode records audio via `MediaRecorder` and needs microphone
  permission plus a secure context (HTTPS, or `localhost` which is treated
  as secure) — the mic button auto-disables itself if the browser doesn't
  support `MediaRecorder` at all.
- Talk mode waits ~1.8s of detected silence (via a Web Audio RMS level
  check) before treating the question as finished, so it's a threshold
  guess, not true end-of-speech detection — a long mid-question pause could
  still finalize early.
- Groq's free tier is rate-limited, not unlimited — fine for a prototype or
  demo, could throttle under sustained heavy use.
