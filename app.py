import os
import logging
import torch
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ─────────────────────────────────────────────
# MODEL LOADING  (once at startup, not per request)
# ─────────────────────────────────────────────
MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

tokenizer = None
model = None

logger.info("=== MODEL LOADING START ===")
logger.info("Loading tokenizer and model for: %s", MODEL_NAME)

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.eval()          # inference mode – disables dropout etc.
    logger.info("=== MODEL LOADING SUCCESS ===")
except Exception:
    # logging.exception prints the full traceback – critical for Render debugging
    logger.exception("=== MODEL LOADING ERROR – full traceback below ===")
    tokenizer = None
    model = None

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
MAX_INPUT_TOKENS = 1024   # DistilBART's hard limit
MAX_INPUT_CHARS  = 15_000
MIN_WORDS        = 20


def split_text(text, max_words=350):
    """Split long text into word-count-bounded chunks."""
    words = text.split()
    return [
        " ".join(words[i : i + max_words])
        for i in range(0, len(words), max_words)
    ]


def summarize_chunk(chunk, max_length=130, min_length=30):
    """
    Tokenize one chunk, generate a summary, and decode it.
    Always runs inside torch.no_grad() to save memory.
    """
    inputs = tokenizer(
        chunk,
        return_tensors="pt",
        max_length=MAX_INPUT_TOKENS,
        truncation=True,         # safely truncate if chunk still too long
        padding=False,
    )

    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_length=max_length,
            min_length=min_length,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/summarize", methods=["POST"])
def summarize():
    # ── 1. Model health check ──────────────────
    if model is None or tokenizer is None:
        logger.error("SUMMARIZATION REQUEST ERROR – model not loaded.")
        return jsonify({
            "success": False,
            "error": (
                "The summarization model failed to load on server startup. "
                "Please check the server logs for the full error traceback."
            ),
        }), 500

    # ── 2. Parse & validate request ───────────
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"success": False, "error": "Invalid request. JSON body with 'text' key required."}), 400

    text = data["text"].strip()

    if not text:
        return jsonify({"success": False, "error": "Please provide some text to summarize."}), 400

    if len(text) > MAX_INPUT_CHARS:
        return jsonify({
            "success": False,
            "error": f"Text is too long. Please limit to approximately {MAX_INPUT_CHARS:,} characters.",
        }), 400

    original_words = len(text.split())
    if original_words < MIN_WORDS:
        return jsonify({
            "success": False,
            "error": f"Text is too short to summarize. Please provide at least {MIN_WORDS} words.",
        }), 400

    # ── 3. Summarise ──────────────────────────
    try:
        chunks = split_text(text, max_words=350)
        summarized_chunks = []

        for i, chunk in enumerate(chunks):
            chunk_word_count = len(chunk.split())
            max_len = min(130, max(30, int(chunk_word_count * 0.6)))
            min_len = min(30, max(10, int(chunk_word_count * 0.2)))

            logger.info("Summarising chunk %d/%d (%d words)...", i + 1, len(chunks), chunk_word_count)
            chunk_summary = summarize_chunk(chunk, max_length=max_len, min_length=min_len)
            if chunk_summary:
                summarized_chunks.append(chunk_summary)

        final_summary = " ".join(summarized_chunks).strip()
        summary_words = len(final_summary.split())

        logger.info(
            "Summarisation complete: %d words -> %d words", original_words, summary_words
        )

        return jsonify({
            "success": True,
            "summary": final_summary,
            "original_words": original_words,
            "summary_words": summary_words,
        })

    except Exception:
        logger.exception("SUMMARIZATION REQUEST ERROR – full traceback below")
        return jsonify({
            "success": False,
            "error": "Unable to generate the summary. Please try again.",
        }), 500


# ─────────────────────────────────────────────
# Entry-point (Gunicorn calls app directly)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
