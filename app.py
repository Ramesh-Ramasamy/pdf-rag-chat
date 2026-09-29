import os

import streamlit as st

import rag
from rag import VectorIndex, answer_stream, build_chunks, gemini_embedder

st.set_page_config(page_title="PDF RAG Chat", page_icon="📄", layout="wide")
st.title("📄 Chat with your PDFs")
st.caption("Retrieval-Augmented Generation with Gemini embeddings, cosine search and cited answers.")


def get_key() -> str | None:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY")


def get_secret(name: str) -> str | None:
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name)


groq_key = get_secret("GROQ_API_KEY")
api_key = get_key() or st.sidebar.text_input("Gemini API key", type="password")
if not api_key:
    st.info("Add GEMINI_API_KEY in app secrets, or paste a key in the sidebar (free at aistudio.google.com).")
    st.stop()

if "index" not in st.session_state:
    st.session_state.index = VectorIndex(gemini_embedder(api_key))
    st.session_state.indexed = set()
    st.session_state.messages = []

with st.sidebar:
    st.header("Documents")
    files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    top_k = st.slider("Chunks to retrieve", 2, 10, 5)
    for f in files or []:
        if f.name not in st.session_state.indexed:
            with st.spinner(f"Indexing {f.name}..."):
                chunks = build_chunks(f.name, f.getvalue())
                if chunks:
                    st.session_state.index.add(chunks)
                    st.session_state.indexed.add(f.name)
                else:
                    st.warning(f"No extractable text in {f.name} (scanned PDF?)")
    st.write(f"Indexed chunks: **{len(st.session_state.index.chunks)}**")
    st.caption(
        f"Build {rag.BUILD} | chat: {'Groq ' + rag.GROQ_MODELS[0] if groq_key else 'Gemini ' + rag.CHAT_MODEL}"
    )

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if q := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        hits = st.session_state.index.search(q, top_k)
        if not hits:
            reply = "Please upload a PDF first."
            st.markdown(reply)
        else:
            try:
                reply = st.write_stream(answer_stream(api_key, q, hits, groq_key))
            except Exception as e:
                reply = f"Error from the model API (models tried: {rag.LAST_MODELS_TRIED}): {e}"
                st.error(reply)
            with st.expander("Sources"):
                for n, (c, s) in enumerate(hits, 1):
                    st.markdown(f"**[{n}] {c.source}, page {c.page}** (score {s:.2f})")
                    st.caption(c.text[:400])
    st.session_state.messages.append({"role": "assistant", "content": reply})
