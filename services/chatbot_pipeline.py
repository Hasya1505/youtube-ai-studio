"""
ChatBot Pipeline (Pipeline 2) – URL ➜ Local FastEmbed ➜ Cached FAISS ➜ RAG Q&A.
Completely independent of the Summary pipeline.
"""

import json
import os
import logging

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings

from services.llm_factory import get_chat_model, invoke_with_retry

logger = logging.getLogger(__name__)

TRANSCRIPT_CACHE_DIR = os.path.join("cache", "transcripts")
FAISS_CACHE_DIR = os.path.join("cache", "faiss_indexes")
os.makedirs(FAISS_CACHE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Singleton local embeddings (zero API quota)
# ---------------------------------------------------------------------------

_EMBED_SINGLETON = None


def get_local_embeddings() -> FastEmbedEmbeddings:
    """Return a singleton FastEmbed instance (BAAI/bge-small-en-v1.5, local ONNX)."""
    global _EMBED_SINGLETON
    if _EMBED_SINGLETON is None:
        logger.info("[EMBED] Initializing local FastEmbed model (CPU/ONNX) …")
        _EMBED_SINGLETON = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return _EMBED_SINGLETON


# ---------------------------------------------------------------------------
# In-memory FAISS cache
# ---------------------------------------------------------------------------

INDEX_MEMORY_CACHE: dict[str, FAISS] = {}


# ---------------------------------------------------------------------------
# Build / Load FAISS index
# ---------------------------------------------------------------------------

def build_or_load_vector_index(video_id: str) -> dict:
    """
    Build (or load from disk) a FAISS vector index for the given video.
    Uses local FastEmbed — 0 API calls consumed.
    """
    index_dir = os.path.join(FAISS_CACHE_DIR, video_id)
    embeddings = get_local_embeddings()

    # --- Cache HIT (disk) ---
    if os.path.exists(index_dir):
        vector_store = FAISS.load_local(
            index_dir, embeddings, allow_dangerous_deserialization=True
        )
        INDEX_MEMORY_CACHE[video_id] = vector_store
        chunk_count = vector_store.index.ntotal
        logger.info("[FAISS] INDEX HIT: %s (%d chunks)", video_id, chunk_count)
        return {
            "video_id": video_id,
            "chunk_count": chunk_count,
            "index_cached": True,
        }

    # --- Cache MISS: build from transcript ---
    transcript_path = os.path.join(TRANSCRIPT_CACHE_DIR, f"{video_id}.json")
    if not os.path.exists(transcript_path):
        raise ValueError(
            "Transcript not loaded yet. Please load the video first on the Load page."
        )

    with open(transcript_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    transcript_text = data["transcript"]

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    docs = splitter.create_documents([transcript_text])

    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(index_dir)
    INDEX_MEMORY_CACHE[video_id] = vector_store

    logger.info("[FAISS] INDEX BUILT & SAVED: %s (%d chunks)", video_id, len(docs))
    return {
        "video_id": video_id,
        "chunk_count": len(docs),
        "index_cached": False,
    }


# ---------------------------------------------------------------------------
# RAG Q&A
# ---------------------------------------------------------------------------

_RAG_PROMPT = PromptTemplate(
    template=(
        "You are a precise YouTube video assistant.\n\n"
        "Answer the user's question using ONLY the transcript excerpts provided below.\n\n"
        "* Use clear formatting (bullet points and bold keywords where helpful).\n"
        "* Do NOT invent external facts or guess beyond the transcript.\n"
        "* If the transcript excerpts do not contain the answer, clearly state: "
        "\"The video transcript does not mention this information.\"\n\n"
        "Transcript Context:\n{context}\n\n"
        "User Question: {question}\n"
        "Answer:"
    ),
    input_variables=["context", "question"],
)


def ask_video_question(
    video_id: str,
    question: str,
    provider: str = "gemini",
    custom_api_key: str | None = None,
) -> dict:
    """
    Answer a user question grounded in the video's FAISS-indexed transcript chunks.
    Retrieves top-4 chunks (~1 200 tokens total) — safe for all free tiers.
    """
    # Ensure the index is in memory (auto-load from disk if needed)
    if video_id not in INDEX_MEMORY_CACHE:
        index_dir = os.path.join(FAISS_CACHE_DIR, video_id)
        if not os.path.exists(index_dir):
            raise ValueError(
                "Vector index not built yet. Please initialize the chatbot first."
            )
        embeddings = get_local_embeddings()
        INDEX_MEMORY_CACHE[video_id] = FAISS.load_local(
            index_dir, embeddings, allow_dangerous_deserialization=True
        )

    vector_store = INDEX_MEMORY_CACHE[video_id]
    retrieved_docs: list[Document] = vector_store.similarity_search(question, k=4)
    context = "\n\n---\n\n".join(doc.page_content for doc in retrieved_docs)

    llm = get_chat_model(provider, custom_api_key=custom_api_key)
    model_name = llm.model_name if hasattr(llm, "model_name") else str(llm.model)
    parser = StrOutputParser()
    chain = _RAG_PROMPT | llm | parser

    answer = invoke_with_retry(chain, {"context": context, "question": question})

    return {
        "answer": answer,
        "sources": [doc.page_content for doc in retrieved_docs],
        "provider": provider,
        "model_used": model_name,
    }
