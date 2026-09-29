# PDF RAG Chat

**Live demo: https://pdf-rag-ch-pf3d2tutvafmixw78zlwqg.streamlit.app**
(free hosting: if the app has been idle it may show a "wake up" button, click it and wait a few seconds)

Chat with your PDFs. A small, production-style Retrieval-Augmented Generation (RAG) app built with Streamlit, Gemini embeddings and a free LLM (Groq Llama, with Gemini as fallback). Every answer cites the source file and page.

Detailed notes for this project: [INTERVIEW.md](INTERVIEW.md)

## How it works

1. **Ingest**: PDFs are parsed page by page (`pypdf`) and split into overlapping chunks (about 900 characters, 150 overlap).
2. **Embed**: chunks are embedded with Gemini `gemini-embedding-001`.
3. **Retrieve**: the question is embedded and matched by cosine similarity (NumPy) to find the top-k chunks.
4. **Generate**: the LLM answers using only the retrieved context and cites sources like `[1]`, with page numbers.
5. **Resilience**: transient errors (429/503) are retried with exponential backoff. If a model is retired or unavailable, the app tries fallback models and discovers currently available models from the provider API.

```
PDF -> chunks -> Gemini embeddings -> in-memory vector index
question -> embedding -> cosine top-k -> grounded prompt -> Groq Llama (fallback: Gemini) -> cited answer
```

## Run locally

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key    # free: https://aistudio.google.com/apikey
export GROQ_API_KEY=your_key      # optional, free: https://console.groq.com/keys
streamlit run app.py
```

## Deploy free (Streamlit Community Cloud)

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Create app, choose this repo, branch `main`, main file `app.py`.
3. In **Advanced settings > Secrets** add:
   ```toml
   GEMINI_API_KEY = "your_gemini_key"
   GROQ_API_KEY = "your_groq_key"
   ```
4. Deploy. Never commit keys to the repo.

## Configuration (environment variables or Streamlit secrets)

| Name | Purpose | Default |
|---|---|---|
| `GEMINI_API_KEY` | embeddings and fallback chat | required |
| `GROQ_API_KEY` | primary chat provider | optional |
| `GROQ_MODELS` | comma-separated Groq models | `llama-3.3-70b-versatile,llama-3.1-8b-instant` |
| `GEMINI_CHAT_MODEL` | Gemini chat model | `gemini-3.8-flash` |
| `GEMINI_FALLBACK_MODELS` | Gemini backups | `gemini-3.5-flash-lite` |

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Design notes and limitations

- The vector index lives in memory per session: simple and free. For persistence, swap `VectorIndex` for pgvector (Supabase or Neon free tier).
- The embedder is injected, so retrieval logic is unit-tested without network calls.
- Scanned PDFs have no extractable text (OCR is a possible next step).
- Possible improvements: hybrid keyword and vector search, a reranker, answer-quality evaluation, authentication and rate limiting.
