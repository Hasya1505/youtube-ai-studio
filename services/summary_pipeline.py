"""
Summary Pipeline (Pipeline 1) – URL ➜ Structured Summary.
Operates directly on the cached transcript text.
Never imports FAISS, FastEmbed, or text-embedding modules.
"""

import json
import os
import time
import logging

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from services.llm_factory import get_chat_model, invoke_with_retry

logger = logging.getLogger(__name__)

TRANSCRIPT_CACHE_DIR = os.path.join("cache", "transcripts")

# ---------------------------------------------------------------------------
# Prompt templates per style
# ---------------------------------------------------------------------------

_STYLE_INSTRUCTIONS = {
    "executive": (
        "Produce a structured Executive Summary with the following sections:\n"
        "### Overview\nA concise paragraph capturing the video's purpose and scope.\n\n"
        "### Top 5 Key Takeaways\nNumbered list of the five most important insights.\n\n"
        "### Core Conclusion\nThe single overarching message or call-to-action from the video."
    ),
    "detailed_notes": (
        "Produce detailed Structured Study Notes with the following sections:\n"
        "### Topic Overview\nA paragraph introducing the subject matter.\n\n"
        "### Section-by-Section Notes\nChronological notes broken into logical sections "
        "with sub-headings, capturing key arguments, examples, and data points.\n\n"
        "### Key Concepts & Definitions\nBullet list of important terms and their definitions.\n\n"
        "### Summary Takeaways\nBrief numbered list of the main lessons."
    ),
    "bullet_takeaways": (
        "Produce concise, scannable Actionable Bullet-Point Takeaways grouped by theme.\n"
        "Use bold keywords at the start of each bullet. "
        "Organize bullets under thematic sub-headings (###). "
        "Focus on practical, actionable insights the viewer can apply immediately."
    ),
}

_SUMMARY_PROMPT = PromptTemplate(
    template=(
        "You are an expert content analyst. Summarize the following YouTube video "
        "transcript according to the instructions below.\n\n"
        "--- FORMAT INSTRUCTIONS ---\n{style_instructions}\n\n"
        "--- TRANSCRIPT ---\n{transcript}\n\n"
        "Summary:"
    ),
    input_variables=["style_instructions", "transcript"],
)

_SEGMENT_SUMMARY_PROMPT = PromptTemplate(
    template=(
        "Summarize the following transcript segment concisely, preserving all "
        "key facts, arguments, and examples:\n\n{segment}\n\nSegment Summary:"
    ),
    input_variables=["segment"],
)

_SYNTHESIS_PROMPT = PromptTemplate(
    template=(
        "You are an expert content analyst. Below are summaries of consecutive "
        "segments from a single YouTube video. Synthesize them into one cohesive "
        "summary following the format instructions.\n\n"
        "--- FORMAT INSTRUCTIONS ---\n{style_instructions}\n\n"
        "--- SEGMENT SUMMARIES ---\n{combined_summaries}\n\n"
        "Final Summary:"
    ),
    input_variables=["style_instructions", "combined_summaries"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_cached_transcript(video_id: str) -> str:
    """Load the transcript text from the disk cache."""
    cache_path = os.path.join(TRANSCRIPT_CACHE_DIR, f"{video_id}.json")
    if not os.path.exists(cache_path):
        raise ValueError(
            "Transcript not loaded yet. Please load the video first on the Load page."
        )
    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["transcript"]


def _chunk_text(text: str, max_chars: int = 24000) -> list[str]:
    """Split text into segments of roughly *max_chars* on word boundaries."""
    words = text.split()
    segments: list[str] = []
    current: list[str] = []
    current_len = 0

    for word in words:
        wl = len(word) + 1  # +1 for space
        if current_len + wl > max_chars and current:
            segments.append(" ".join(current))
            current = []
            current_len = 0
        current.append(word)
        current_len += wl

    if current:
        segments.append(" ".join(current))
    return segments


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_video_summary(
    video_id: str,
    provider: str = "gemini",
    summary_style: str = "executive",
    custom_api_key: str | None = None,
) -> dict:
    """
    Generate a structured summary of the video.

    Token-Safe Strategy:
      • Single-Pass: used when provider == "gemini" OR transcript ≤ 28 000 chars.
      • Chunked Map-Reduce: used when provider == "groq" AND transcript > 28 000 chars
        to stay under Groq's 12 000 TPM free-tier limit.
    """
    transcript = _load_cached_transcript(video_id)
    style_instructions = _STYLE_INSTRUCTIONS.get(summary_style, _STYLE_INSTRUCTIONS["executive"])

    llm = get_chat_model(provider, custom_api_key=custom_api_key)
    model_name = llm.model_name if hasattr(llm, "model_name") else str(llm.model)
    parser = StrOutputParser()

    use_map_reduce = provider == "groq" and len(transcript) > 28000

    if not use_map_reduce:
        # ----- Single-Pass Mode -----
        chain = _SUMMARY_PROMPT | llm | parser
        result = invoke_with_retry(
            chain,
            {"style_instructions": style_instructions, "transcript": transcript},
        )
        mode = "single_pass"
        logger.info("[SUMMARY] Single-pass complete for %s via %s", video_id, provider)
    else:
        # ----- Chunked Map-Reduce Mode -----
        segments = _chunk_text(transcript, max_chars=24000)
        logger.info(
            "[SUMMARY] Map-Reduce: %d segments for %s via %s",
            len(segments), video_id, provider,
        )

        segment_chain = _SEGMENT_SUMMARY_PROMPT | llm | parser
        segment_summaries: list[str] = []

        for idx, segment in enumerate(segments):
            logger.info("[SUMMARY] Summarizing segment %d/%d …", idx + 1, len(segments))
            seg_result = invoke_with_retry(segment_chain, {"segment": segment})
            segment_summaries.append(seg_result)
            if idx < len(segments) - 1:
                time.sleep(2.5)  # Cool-down for Groq TPM window

        combined = "\n\n---\n\n".join(segment_summaries)
        synthesis_chain = _SYNTHESIS_PROMPT | llm | parser
        # Cool-down before final synthesis call
        time.sleep(2.5)
        result = invoke_with_retry(
            synthesis_chain,
            {"style_instructions": style_instructions, "combined_summaries": combined},
        )
        mode = "map_reduce"
        logger.info("[SUMMARY] Map-Reduce synthesis complete for %s", video_id)

    return {
        "summary": result,
        "provider": provider,
        "model_used": model_name,
        "mode": mode,
    }
