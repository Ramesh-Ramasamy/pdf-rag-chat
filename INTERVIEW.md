# Interview notes: PDF RAG Chat

Live demo: https://pdf-rag-ch-pf3d2tutvafmixw78zlwqg.streamlit.app

## 30-second pitch

I built and deployed a RAG app where you upload PDFs and ask questions. It answers only from your documents and cites the file and page. RAG (Retrieval-Augmented Generation) means I first retrieve the relevant passages, then have the LLM answer using only those, instead of trusting the model's memory. It uses Python, Streamlit, Gemini embeddings and a free LLM (Groq Llama with Gemini fallback), and is hosted free on Streamlit Community Cloud straight from GitHub, with API keys in secrets.

## Architecture walkthrough

1. **Ingest**: extract text page by page with `pypdf`; split into ~900 character chunks with 150 overlap.
2. **Embed**: each chunk becomes a vector via Gemini `gemini-embedding-001` (document task type). The question uses the query task type.
3. **Index**: vectors are normalised and stored in a NumPy matrix, one per session.
4. **Retrieve**: cosine similarity (a dot product on normalised vectors) picks the top-k chunks.
5. **Generate**: a grounded prompt contains the chunks and asks the model to answer only from them, cite `[n]`, and say so if the answer is missing. The answer streams to the UI.
6. **Verify**: the UI lists the source chunks with file, page and score.

## Problems I hit and how I solved them (tell these as stories)

1. **Retired model (404).** The chat model I first chose was no longer available to new API keys. I read the error, moved to the recommended model, and made model names configurable via environment variables.
2. **Overload errors (503/429) on the free tier.** I added retries with exponential backoff (1s, 2s, 4s). Retries only happen before any text has streamed, so a half-finished answer is never restarted.
3. **Model names going stale twice.** Instead of hardcoding, the app now asks the provider API which models the key can use and tries the newest ones. Failure order: Groq models, then Gemini models, then discovered models.
4. **Swapping providers.** When Gemini chat was unreliable I added Groq (OpenAI-compatible streaming API) behind the same interface, kept Gemini for embeddings, and used Gemini as the fallback. Because the code is provider-agnostic, this took one function, not a rewrite.
5. **Deployment confusion.** The running app was on old code after a fix. I added a build label in the sidebar so I can see exactly which version is deployed. Lesson: make deployments observable.
6. **Secrets.** API keys live in Streamlit secrets and environment variables, never in the repo.

## Likely questions and answers

**Why RAG instead of fine-tuning?**
RAG is cheaper, needs no training, reflects document changes immediately, and gives citations. Fine-tuning teaches style or skills, not fresh facts.

**Why chunk with overlap?**
So a sentence or idea cut at a chunk boundary still appears whole in a neighbouring chunk, which protects retrieval quality.

**How did you choose chunk size and top-k?**
900 characters is roughly a paragraph: small enough to be specific, big enough to keep context. Top-k is 5 by default and adjustable in the UI. With more time I would tune both on a labelled question set.

**Why cosine similarity?**
It compares direction (meaning) rather than magnitude. With normalised vectors it is just a dot product, so NumPy does it in one matrix multiply.

**How do you reduce hallucinations?**
A grounded prompt ("answer only from context, otherwise say you could not find it"), low temperature, retrieval of only relevant chunks, and showing sources so users can verify.

**How would you make the index persistent and scale it?**
Store embeddings in Postgres with pgvector (Supabase or Neon free tier) or a vector DB, ingest documents asynchronously, cache embeddings by file hash, and put a proper backend API behind the UI.

**How would you evaluate quality?**
Build a small set of questions with expected source pages. Measure retrieval recall@k (is the right chunk in the top k?) and answer faithfulness (is the answer supported by the retrieved text?). Track these over changes to chunking, embeddings and prompts.

**How would you improve retrieval?**
Hybrid search (BM25 keyword plus vectors), a reranker on the top 20 results, query rewriting, and metadata filters.

**What are the limitations?**
In-memory index (lost when the session ends), no OCR for scanned PDFs, no authentication or rate limiting, and dependence on free-tier quotas.

**How do you handle failures from the LLM API?**
Classify errors: transient ones (429, 503) are retried with backoff; permanent ones (404 model retired) skip to the next model. If a stream has already started, I raise instead of retrying so users never see a restarted answer.

**How is it deployed and how are secrets handled?**
Streamlit Community Cloud deploys from the GitHub `main` branch automatically. Keys are stored in the platform's secrets, read via `st.secrets` or environment variables, and are not in the repository.

**What would you do differently in production?**
Persistent vector store, auth, per-user quotas, structured logging and tracing, an automated evaluation pipeline in CI, and a proper API layer separate from the UI.

**Why unit-test with a fake embedder?**
Injecting the embedder makes retrieval logic deterministic and fast to test without network calls or API costs.
