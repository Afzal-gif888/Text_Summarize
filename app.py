import os
import logging
from flask import Flask, request, jsonify, render_template
from transformers import pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Load the model globally at startup
logging.info("Loading Hugging Face summarization model...")
try:
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    summarizer = None

def split_text(text, max_words=350):
    """Split text into chunks of max_words to handle long inputs."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)
    return chunks

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    if not summarizer:
        return jsonify({"success": False, "error": "Model failed to load on server startup."}), 500

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"success": False, "error": "Invalid request."}), 400

    text = data['text'].strip()
    
    if not text:
        return jsonify({"success": False, "error": "Please provide some text to summarize."}), 400
        
    if len(text) > 15000:
        return jsonify({"success": False, "error": "Text is too long. Please limit to approximately 15,000 characters."}), 400

    original_words = len(text.split())
    if original_words < 20:
        return jsonify({"success": False, "error": "Text is too short to summarize. Please provide at least 20 words."}), 400

    try:
        # Process chunks
        chunks = split_text(text, max_words=350)
        summarized_chunks = []
        
        for chunk in chunks:
            chunk_word_count = len(chunk.split())
            max_len = min(130, max(30, int(chunk_word_count * 0.6)))
            min_len = min(30, int(chunk_word_count * 0.2))
            
            # Generate summary for chunk
            res = summarizer(chunk, max_length=max_len, min_length=min_len, do_sample=False)
            if res and len(res) > 0:
                summarized_chunks.append(res[0]['summary_text'])
                
        final_summary = " ".join(summarized_chunks).strip()
        summary_words = len(final_summary.split())
        
        return jsonify({
            "success": True,
            "summary": final_summary,
            "original_words": original_words,
            "summary_words": summary_words
        })
        
    except Exception as e:
        # Log the internal error for debugging
        logging.error(f"Inference error: {e}")
        # Return generic error to frontend
        return jsonify({
            "success": False, 
            "error": "Unable to generate the summary. Please try again."
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
