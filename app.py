import os
import gc
import logging
import torch
from typing import Optional, Any
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Limit PyTorch CPU thread count to 1 to reduce RAM footprint on cloud containers
torch.set_num_threads(1)

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
# MODEL LOADING  (t5-small: ~60M params, ~68MB peak RAM)
# ─────────────────────────────────────────────
MODEL_NAME = "t5-small"

tokenizer: Optional[Any] = None
model: Optional[Any] = None


def load_model() -> bool:
    """Lazily load the model on server startup or first request."""
    global tokenizer, model
    if model is not None and tokenizer is not None:
        return True

    logger.info("=== MODEL LOADING START: %s ===", MODEL_NAME)
    try:
        tokenizer_cls: Any = AutoTokenizer
        model_cls: Any = AutoModelForSeq2SeqLM

        loaded_tok = tokenizer_cls.from_pretrained(MODEL_NAME)
        loaded_mod = model_cls.from_pretrained(
            MODEL_NAME,
            low_cpu_mem_usage=True,
        )
        if loaded_mod is not None and hasattr(loaded_mod, "eval"):
            loaded_mod.eval()

        tokenizer = loaded_tok
        model = loaded_mod
        gc.collect()
        logger.info("=== MODEL LOADING SUCCESS ===")
        return True
    except Exception:
        logger.exception("=== MODEL LOADING ERROR – full traceback below ===")
        tokenizer = None
        model = None
        return False


# Attempt eager model loading at startup
load_model()

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
MAX_INPUT_TOKENS = 256
MAX_INPUT_CHARS  = 15_000
MIN_WORDS        = 20


def split_text(text: str, max_words: int = 250):
    """Split long text into word-count-bounded chunks."""
    words = text.split()
    return [
        " ".join(words[i : i + max_words])
        for i in range(0, len(words), max_words)
    ]


def summarize_chunk(chunk: str, max_length: int = 100, min_length: int = 20) -> str:
    """
    Tokenize one chunk, generate a summary, and decode it.
    Always runs inside torch.no_grad() with num_beams=1 to save memory.
    """
    assert tokenizer is not None, "tokenizer must be loaded"
    assert model is not None, "model must be loaded"

    prompt_text = "summarize: " + chunk
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        max_length=MAX_INPUT_TOKENS,
        truncation=True,
        padding=False,
    )

    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_length=max_length,
            min_length=min_length,
            num_beams=1,           # Greedy decoding: keeps peak RAM under 500MB
            do_sample=False,
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
    # Ensure model is loaded (lazy fallback)
    if not load_model():
        logger.error("SUMMARIZATION REQUEST ERROR – model not loaded.")
        return jsonify({
            "success": False,
            "error": (
                "The summarization model failed to load. "
                "Please check the server logs for details."
            ),
        }), 500

    # Parse & validate request
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

    # Summarise
    try:
        chunks = split_text(text, max_words=250)
        summarized_chunks = []

        for i, chunk in enumerate(chunks):
            chunk_word_count = len(chunk.split())
            max_len = min(100, max(25, int(chunk_word_count * 0.6)))
            min_len = min(25, max(10, int(chunk_word_count * 0.2)))

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
