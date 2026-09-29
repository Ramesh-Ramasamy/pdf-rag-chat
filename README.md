# PDF RAG Chat

Chat with your PDFs. A small, production-style Retrieval-Augmented Generation app built with Streamlit and Google Gemini (free tier).

## How it works

1. **Ingest**: PDFs are parsed page by page (`pypdf`) and split into overlapping chunks.
2. **Embed**: chunks are embedded with Gemini `gemini-embedding-001`.
3. **Retrieve**: the question is embedded and matched by cosine similarity (NumPy) to find the top-k chunks.
4. **Generate**: Gemini answers using only the retrieved context and cites sources like `[1]`, with page numbers.

## Run locally

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key   # free key: https://aistudio.google.com/apikey
streamlit run app.py
```

## Deploy free (Streamlit Community Cloud)

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Create app, choose this repo, branch `main`, main file `app.py`.
3. In **Advanced settings > Secrets** add: `GEMINI_API_KEY = "your_key"`.
4. Deploy. Never commit the key to the repo.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

CI runs the tests on every push via GitHub Actions.

## Design notes

- In-memory vector index per session: simple, no infrastructure cost. For persistence, swap `VectorIndex` for pgvector (Supabase/Neon free tier).
- The embedder is injected, so retrieval logic is unit-tested without network calls.
- Models are configurable via `GEMINI_CHAT_MODEL` and `GEMINI_EMBED_MODEL`.
