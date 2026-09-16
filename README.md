# AI Text Summarizer

A complete, production-quality mini project to summarize long text using AI.

## Features
- **AI-Powered Summarization:** Uses Hugging Face's `sshleifer/distilbart-cnn-12-6` model to generate concise summaries.
- **Large Text Support:** Automatically chunks large text to bypass transformer input limits.
- **Modern UI/UX:** Clean, responsive, and accessible interface built with HTML, CSS, and Vanilla JS (No React/Tailwind).
- **Live Statistics:** Displays word counts and reduction percentages.

## Technology Stack
- **Frontend:** HTML5, CSS3, Vanilla JavaScript
- **Backend:** Python 3, Flask
- **AI/ML:** Hugging Face Transformers, PyTorch, SentencePiece, NumPy

## Folder Structure
```
Text_Summarize/
└── ai-text-summarizer/
    ├── app.py                 # Main Flask application and API
    ├── requirements.txt       # Python dependencies
    ├── README.md              # Project documentation
    ├── templates/
    │   └── index.html         # Main HTML file
    └── static/
        ├── css/
        │   └── style.css      # Custom styling
        └── js/
            └── script.js      # Frontend logic and API calls
```

## Deployment Instructions

### LOCAL WINDOWS:
Run these commands from `C:\Users\afzal\Desktop\projects\Text_Summarize`

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r ai-text-summarizer\requirements.txt
cd ai-text-summarizer
python app.py
```
*Note: If you run `python app.py` from within `ai-text-summarizer` directly, you must activate the root `venv` first.*

### PRODUCTION:
Run these commands from `ai-text-summarizer/` where `app.py` is located.

```bash
pip install -r requirements.txt
gunicorn app:app
```

## How Hugging Face Works Here
This project uses the `pipeline("summarization")` from the `transformers` library. The model is loaded exactly once when the Flask server starts, so it can be reused efficiently across multiple API requests without needing to reload the heavy model into memory every time. This is especially important for production servers.
