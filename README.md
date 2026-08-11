# HOVERAir Manual Assistant (RAG demo)

A chatbot that answers questions about HOVERAir's drone lineup (X1, X1 PRO,
X1 PROMAX, AQUA) by retrieving the relevant section(s) of the official
manuals and having a locally-hosted LLM answer strictly from that text,
with citations back to the sections used. This is the prototype described
in the opportunity report: HOVERAir currently has no smart support
assistant, only static PDFs and a contact form.

Everything runs locally through **Ollama** — no API key, no per-request
cost, no data leaving your machine.

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
3. `app.py` is a Flask server. For each question it embeds the question,
   retrieves the top matching manual sections, and streams an answer
   token-by-token from `llama3.1:8b` (via Ollama) with instructions to
   answer only from those excerpts and never from outside knowledge.
   Follow-up questions carry conversation history for context. Price
   questions get live pricing pulled from HOVERAir's Shopify store
   (`price_fetcher.py`) injected into the prompt. Common/repeated
   first-time questions are served instantly from a local FAQ cache
   (`faq_cache.py`) instead of re-running the model.
4. `static/index.html` offers two modes, switched with a toggle at the top:
   - **Chat with the Agent** — type a question, get a written answer with
     source citation chips.
   - **Talk to the Agent** — tap the mic button and ask out loud. The
     browser's built-in Web Speech API transcribes it locally (no server
     round-trip for speech-to-text) and submits it to `/api/chat` tagged
     `mode: "talk"`, which asks for a short, conversational, unformatted
     answer suited to being read out loud via the browser's speech
     synthesis. The full text still appears in the chat transcript (with
     source chips) as a reference.

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

ollama pull llama3.1:8b          # generation model (~4.9 GB, one-time)
ollama pull nomic-embed-text     # embedding model (~274 MB, one-time)

./venv/bin/python build_index.py    # builds index.pkl from manuals/
./venv/bin/python app.py            # serves on http://localhost:5001
```

Ollama itself must be running (`ollama serve`, or the Ollama.app menu bar
app) before `build_index.py` or `app.py` will work.

Then open http://localhost:5001 in your browser.

## Performance

`llama3.1:8b` is genuinely slow on hardware that can't hold the whole model
in GPU/unified memory — 10-70+ seconds per answer is normal, not a bug.
Run `ollama ps` to see the current CPU/GPU split if answers feel unusually
slow.

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

- Local inference is slow relative to a hosted API, especially on hardware
  that can't fit the whole model in GPU/unified memory — 10-70+ seconds
  per answer is expected here.
- Answers are only as good as the manual text included — it doesn't cover
  every edge case (e.g. detailed troubleshooting for specific error
  codes), since HOVERAir doesn't publish that publicly.
- This is a prototype, not a production support tool: no auth, no rate
  limiting, and the dev server binds to every network interface with
  Werkzeug's debugger active — fine for solo local use, not safe to leave
  running on a shared network.
- Talk mode relies on the browser's `SpeechRecognition`/`SpeechSynthesis`
  APIs: works well in Chrome and Edge, is unsupported in Firefox, and is
  partial in Safari. It also requires a secure context (HTTPS, or
  `localhost` which is treated as secure) — the mic button auto-disables
  itself if the browser doesn't support voice input at all.
- Talk mode waits ~1.8s of silence after your last word before treating
  the question as finished (instead of cutting off at the first short
  pause), so it's a `setTimeout` guess, not true end-of-speech detection —
  a long mid-question pause could still finalize early.
