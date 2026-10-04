# HumanizeAI
HumanizeAI is an AI-powered document processing platform that analyzes, rewrites, and improves text while preserving important information and document structure. It supports text and multiple document formats, table-aware processing, AI-likeness analysis, and downloadable outputs through an interactive Streamlit interface powered by Groq.
# ✨ HumanizeAI

**Humanize. Detect. Improve.**

An AI-powered document processing app built with Streamlit and Groq. It rewrites text
to read naturally, estimates how AI-like a text appears, and preserves document
structure, including tables.

## Features
- Paste text or upload DOCX, PDF, TXT, MD, CSV, XLSX, XLS
- Table-aware: table values are never changed
- Numbers in paragraphs are validated after every rewrite
- 8 writing styles (Natural, Academic, Professional, ...)
- AI-likelihood estimate (probabilistic, not proof)
- Detect + Humanize with before/after comparison
- Side-by-side original vs humanized view
- Download as TXT, DOCX, PDF (and XLSX for data files)
- Large documents processed in chunks (no app-imposed word/page limit)

## Run locally
```bash
pip install -r requirements.txt
# create .streamlit/secrets.toml containing:
# GROQ_API_KEY = "your_key_here"
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
1. Push this repo to GitHub.
2. Go to share.streamlit.io, click **Create app**, pick the repo, and set the main file to `app.py`.
3. Under **Advanced settings → Secrets** add: `GROQ_API_KEY = "your_key_here"`
4. Deploy.

## Notes
- Uses Groq's free models. Groq's free-tier rate limits still apply.
- Old `.doc` files are not supported; save as `.docx`.
- Scanned PDFs (images) need OCR and are not supported.
- AI detection is an estimate and should not be treated as proof of authorship.
- Never commit your API key. `.streamlit/secrets.toml` is in `.gitignore`.
