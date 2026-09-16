# AI Text Summarizer

A complete, production-quality mini project using Python, Flask, Hugging Face Transformers, HTML, CSS, and Vanilla JavaScript.

## Features
- Paste a large block of text into an input area
- Generate a concise summary using a pretrained Hugging Face summarization model (`sshleifer/distilbart-cnn-12-6`)
- Automatically chunks large text to bypass model token limits
- Live word counter and character limit validation
- Responsive UI design inspired by modern SaaS applications
- Built entirely without heavy frontend frameworks

## Technology Stack
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Backend**: Python 3, Flask
- **AI**: Hugging Face Transformers, PyTorch

## Project Architecture
```
ai-text-summarizer/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── templates/
│   └── index.html
│
└── static/
    ├── css/
    │   └── style.css
    │
    └── js/
        └── script.js
```

## Installation Instructions

1. **Create a virtual environment (Windows):**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: This might take some time as it installs PyTorch and Transformers.)*

3. **Run the application:**
   ```bash
   python app.py
   ```
   *(Note: The first startup may take time because Hugging Face will download the pretrained model `sshleifer/distilbart-cnn-12-6`. Subsequent runs will use the locally cached model.)*

4. **Open in browser:**
   Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your web browser.