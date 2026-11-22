"""
Transcript Service – YouTube URL parser, caption fetcher, cleaner & disk cache.
Shared by both Pipeline 1 (Summary) and Pipeline 2 (ChatBot RAG).
"""

import json
import os
import re
import logging

from youtube_transcript_api import YouTubeTranscriptApi

logger = logging.getLogger(__name__)

TRANSCRIPT_CACHE_DIR = os.path.join("cache", "transcripts")
os.makedirs(TRANSCRIPT_CACHE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# URL Parsing
# ---------------------------------------------------------------------------

def extract_video_id(url: str) -> str:
    """
    Extract an 11-character YouTube video ID from various URL formats:
      - https://www.youtube.com/watch?v=VIDEO_ID
      - https://youtu.be/VIDEO_ID
      - https://www.youtube.com/shorts/VIDEO_ID
      - https://www.youtube.com/embed/VIDEO_ID
    """
    if not url or not isinstance(url, str):
        raise ValueError("Invalid YouTube URL. Please enter a valid video link.")

    url = url.strip()

    patterns = [
        # Standard watch URL
        r'(?:https?://)?(?:www\.)?youtube\.com/watch\?.*v=([A-Za-z0-9_-]{11})',
        # Shortened URL
        r'(?:https?://)?youtu\.be/([A-Za-z0-9_-]{11})',
        # Shorts URL
        r'(?:https?://)?(?:www\.)?youtube\.com/shorts/([A-Za-z0-9_-]{11})',
        # Embed URL
        r'(?:https?://)?(?:www\.)?youtube\.com/embed/([A-Za-z0-9_-]{11})',
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    raise ValueError("Invalid YouTube URL. Please enter a valid video link.")


# ---------------------------------------------------------------------------
# Transcript Cleaning
# ---------------------------------------------------------------------------

_STAGE_ANNOTATION_RE = re.compile(
    r'\[(?:Music|Applause|Laughter|Cheering|Silence|Inaudible|Background Noise|Foreign)\]',
    re.IGNORECASE,
)


def _clean_transcript_text(raw_text: str) -> str:
    """Strip stage annotations and collapse excess whitespace."""
    text = _STAGE_ANNOTATION_RE.sub('', raw_text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ---------------------------------------------------------------------------
# Fetch / Cache
# ---------------------------------------------------------------------------

def get_or_fetch_transcript(youtube_url: str) -> dict:
    """
    Return a transcript dict for the given YouTube URL.
    Uses disk cache under cache/transcripts/{video_id}.json to avoid
    redundant network calls.
    """
    video_id = extract_video_id(youtube_url)
    cache_path = os.path.join(TRANSCRIPT_CACHE_DIR, f"{video_id}.json")

    # --- Cache HIT ---
    if os.path.exists(cache_path):
        logger.info("[TRANSCRIPT] CACHE HIT: %s", video_id)
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "video_id": data["video_id"],
            "transcript": data["transcript"],
            "word_count": data["word_count"],
            "estimated_tokens": data["estimated_tokens"],
            "cached": True,
        }

    # --- Cache MISS: fetch from YouTube ---
    try:
        ytt_api = YouTubeTranscriptApi()
        fetched = ytt_api.fetch(video_id)
        raw_text = " ".join(snippet.text for snippet in fetched)
    except Exception as exc:
        error_msg = str(exc).lower()
        if "disabled" in error_msg or "unavailable" in error_msg or "no transcript" in error_msg:
            raise ValueError(
                "Captions are disabled or unavailable for this video."
            ) from exc
        raise ValueError(
            f"Could not fetch transcript: {exc}"
        ) from exc

    cleaned = _clean_transcript_text(raw_text)

    if not cleaned:
        raise ValueError("Captions are disabled or unavailable for this video.")

    word_count = len(cleaned.split())
    estimated_tokens = int(word_count * 1.3)

    payload = {
        "video_id": video_id,
        "youtube_url": youtube_url,
        "transcript": cleaned,
        "word_count": word_count,
        "estimated_tokens": estimated_tokens,
    }

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    logger.info("[TRANSCRIPT] CACHE MISS (Saved): %s", video_id)

    return {
        "video_id": video_id,
        "transcript": cleaned,
        "word_count": word_count,
        "estimated_tokens": estimated_tokens,
        "cached": False,
    }
