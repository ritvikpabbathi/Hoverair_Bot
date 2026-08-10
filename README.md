# HOVERAir X1 Manual Assistant (RAG demo)

A chatbot that answers questions about the HOVERAir X1 drone by retrieving
the relevant section of the official manual and having Claude answer strictly
from that text, with a citation back to the section it used. This is the
prototype described in the opportunity report: HOVERAir currently has no
smart support assistant, only a static PDF and a contact form.

## How it works

1. `manuals/hoverair_x1_manual.txt` holds the official Quick Start Guide, App
   Instructions, and Safety Instructions, organized into `SOURCE:` /
   `SECTION:` blocks.
2. `build_index.py` splits the manual into per-section chunks and builds a
   TF-IDF vector index over them (via scikit-learn). No model download
   needed, and it retrieves well for a manual-sized corpus because
   questions and answers tend to share vocabulary ("firmware", "land",
   "flight mode", etc). If you want to handle more paraphrased/casual
   questions later, swap the `retrieve()` function in `app.py` for real
   sentence embeddings (e.g. sentence-transformers or Voyage AI) — same
   interface, better semantic matching, more setup.
3. `app.py` is a small Flask server. For each question it retrieves the
   top matching sections, then sends only those sections to Claude Haiku
   4.5 with instructions to answer only from that text and cite the
   section used. This keeps answers grounded and keeps the cost tiny
   (a few hundred tokens per question).
4. `static/index.html` offers two modes, switched with a toggle at the top:
   - **Chat with the Agent** - type a question, get a written answer with
     a "Source: ..." citation line and source chips.
   - **Talk to the Agent** - tap the mic button and ask out loud. The
     browser's built-in Web Speech API transcribes it locally (no
     server round-trip for speech-to-text) and submits it to `/api/chat`
     tagged `mode: "talk"`, which asks Claude for a short, conversational,
     unformatted answer (no markdown, no citation line) suited to being
     read out loud via the browser's speech synthesis. The text still
     appears in the chat transcript (with source chips) as a reference.

   Both modes answer strictly from the retrieved manual excerpts and never
   state a fact that isn't in them. If a question isn't covered by the
   manual (e.g. "what's the return policy?"), the agent doesn't just give
   a flat canned refusal — it names the topic it heard, says plainly that
   the manual doesn't cover it, and suggests checking HOVERAir's website
   or support, without guessing at what the actual answer might be.

## Setup

```bash
cd hoverair_bot
pip install -r requirements.txt
python build_index.py          # builds index.pkl from the manual
export ANTHROPIC_API_KEY=sk-ant-...   # get one at console.anthropic.com
python app.py
```

Then open http://localhost:5001 in your browser.

## Cost

Each question costs a fraction of a cent (Claude Haiku 4.5: $1/$5 per
million input/output tokens, and each request is only a few hundred to
~1,000 tokens of manual context). New Anthropic accounts also get a small
amount of free trial credit.

## Extending this to a different product

1. Replace `manuals/hoverair_x1_manual.txt` with a new manual, using the
   same `SOURCE:` / `SECTION:` structure so citations keep working.
2. Re-run `python build_index.py`.
3. Update the product name in `app.py` (`build_prompt`) and the suggested
   questions in `static/index.html`.

## Known limitations (worth knowing before demoing)

- TF-IDF retrieval matches on shared words, not meaning — it won't
  reliably catch a heavily reworded question that shares no vocabulary
  with the manual (e.g. "it won't stop spinning" vs. "propellers do not
  stop after landing" would actually work here since "propellers"/"stop"
  overlap, but a truly unrelated phrasing might miss). This is a known,
  documented tradeoff, not a bug — the fix is swapping in embeddings.
- Answers are only as good as the manual text included. The manual here
  covers flying, the app, firmware updates, and safety; it does not cover
  every edge case (e.g. detailed troubleshooting for specific error
  codes), since HOVERAir doesn't publish that publicly.
- This is a prototype, not a production support tool: no conversation
  history/memory across turns, no rate limiting, no auth.
- Talk mode relies on the browser's `SpeechRecognition`/`SpeechSynthesis`
  APIs: works well in Chrome and Edge, is unsupported in Firefox, and is
  partial in Safari. It also requires a secure context (HTTPS, or
  `localhost` which is treated as secure) — the mic button auto-disables
  itself if the browser doesn't support voice input at all.
- Talk mode waits ~1.8s of silence after your last word before treating
  the question as finished (instead of cutting off at the first short
  pause), so it's a `setTimeout` guess, not true end-of-speech detection —
  a long mid-question pause could still finalize early.
