# 🧠 DocuMind AI — Multimodal Document Intelligence Platform

> A production-ready AI document analysis platform powered by Google Gemini.
> Upload any document, extract insights, chat with your files, and generate study materials.

---

## ✨ Features

- **Multi-format support**: PDF, DOCX, TXT, JPG, PNG (scanned docs via OCR)
- **AI Q&A with citations**: Ask anything, get answers from your documents
- **RAG Search**: Semantic vector embeddings with FAISS (keyword fallback)
- **Study Tools**: Flashcards, quizzes, structured study notes
- **Analytics Dashboard**: Charts, sentiment analysis, entity extraction
- **Export**: PDF and TXT reports, chat history JSON
- **Security**: File validation, text sanitization, prompt injection detection

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**OCR (Ubuntu/Debian):**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-eng poppler-utils
```

**OCR (macOS):**
```bash
brew install tesseract poppler
```

### 2. Configure API Key

```toml
# .streamlit/secrets.toml
GEMINI_API_KEY = "AIza_your_key_here"
```

Get a free key at: https://makersuite.google.com/app/apikey

### 3. Run

```bash
streamlit run app.py
```

---

## ☁️ Deploy to Streamlit Community Cloud

1. Push your code to GitHub (exclude `secrets.toml` via `.gitignore`)
2. Go to https://share.streamlit.io → **New app**
3. Select repo → set main file: `app.py`
4. **Advanced settings → Secrets** → paste `GEMINI_API_KEY = "AIza..."`
5. Click **Deploy!**

The `packages.txt` file handles Tesseract OCR installation automatically on Streamlit Cloud.

---

## 📁 Project Structure

```
multimodal-document-analyzer/
├── app.py                    # Main Streamlit app (all UI pages)
├── requirements.txt          # Python dependencies
├── packages.txt              # System packages (Tesseract OCR)
├── README.md
├── .streamlit/
│   ├── config.toml           # Theme & server config
│   └── secrets.toml          # API keys (never commit this!)
└── utils/
    ├── __init__.py
    ├── ai_utils.py           # All Gemini AI functions
    ├── pdf_utils.py          # PDF & DOCX extraction
    ├── image_utils.py        # Image OCR & captioning
    ├── ocr_utils.py          # Enhanced OCR pipeline
    ├── embeddings.py         # Vector RAG search
    ├── export_utils.py       # PDF/TXT export
    └── security.py           # Validation & sanitization
```

---

## 🔑 API Key Options

| Method | How |
|--------|-----|
| Streamlit Secrets | `.streamlit/secrets.toml`: `GEMINI_API_KEY = "AIza..."` |
| Environment Variable | `export GEMINI_API_KEY="AIza..."` |
| In-App | Toggle "Use My Own API Key" in sidebar |

---

## 🛡️ Security Features

- File type whitelist (PDF, DOCX, TXT, JPG, PNG only)
- 50MB upload size limit
- HTML/script injection removal from extracted text
- Prompt injection pattern detection
- API key hidden from all UI elements
- Session isolation between users

---

## 🤖 Supported Models

| Model | Speed | Cost | Best For |
|-------|-------|------|----------|
| `gemini-1.5-flash` | ⚡ Fast | 💚 Low | Default, everyday use |
| `gemini-1.5-pro` | 🐢 Slower | 🔴 Higher | Complex analysis |
| `gemini-2.0-flash` | ⚡⚡ Fastest | 💚 Lowest | High volume |

Change in Settings → AI Model tab.

---

## 📦 Dependencies

Core:
- `streamlit` — UI framework
- `google-generativeai` — Gemini AI
- `pdfplumber` — PDF extraction
- `python-docx` — DOCX parsing
- `pytesseract` + `Pillow` — OCR
- `plotly` — Charts
- `reportlab` — PDF export

Optional (for semantic search):
- `sentence-transformers` — Embeddings
- `faiss-cpu` — Vector index

---

## 🧩 Troubleshooting

**OCR not working:**
- Install Tesseract: `sudo apt install tesseract-ocr`
- Or disable OCR in Settings and use AI vision instead

**API errors:**
- Check API key is valid (starts with `AIza`)
- Ensure Gemini API is enabled in Google Cloud Console
- Check rate limits (free tier: 15 RPM)

**PDF extraction empty:**
- PDF may be scanned → enable OCR in Settings
- Try a different PDF that has selectable text

**Deploy fails on Streamlit Cloud:**
- Check `packages.txt` has `tesseract-ocr`
- Check `requirements.txt` versions are compatible
- View logs in Streamlit Cloud dashboard

---

## 📄 License

MIT License — Free to use, modify, and deploy.

---

Built with ❤️ using Streamlit + Google Gemini
