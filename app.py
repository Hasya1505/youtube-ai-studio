import logging
import os

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify

from services.transcript_service import get_or_fetch_transcript
from services.summary_pipeline import generate_video_summary
from services.chatbot_pipeline import build_or_load_vector_index, ask_video_question
from services.llm_factory import RateLimitException

# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)

app = Flask(__name__)

# Ensure cache directories exist
os.makedirs(os.path.join("cache", "transcripts"), exist_ok=True)
os.makedirs(os.path.join("cache", "faiss_indexes"), exist_ok=True)


# ---------------------------------------------------------------------------
# HTML Page Routes
# ---------------------------------------------------------------------------

@app.route("/")
def page_load():
    return render_template("load.html")


@app.route("/summarize")
def page_summarize():
    return render_template("summarize.html")


@app.route("/chatbot")
def page_chatbot():
    return render_template("chatbot.html")


# ---------------------------------------------------------------------------
# REST API Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/load")
def api_load():
    """Load & cache a YouTube transcript."""
    data = request.get_json(silent=True) or {}
    youtube_url = (data.get("youtube_url") or "").strip()

    if not youtube_url:
        return jsonify({"error": "Please provide a YouTube URL."}), 400

    try:
        result = get_or_fetch_transcript(youtube_url)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logging.exception("Unexpected error in /api/load")
        return jsonify({"error": f"Server error: {exc}"}), 500


@app.post("/api/summarize")
def api_summarize():
    """Generate a structured video summary (Pipeline 1)."""
    data = request.get_json(silent=True) or {}
    video_id = (data.get("video_id") or "").strip()
    provider = (data.get("provider") or "gemini").strip().lower()
    summary_style = (data.get("summary_style") or "executive").strip().lower()
    api_key = (data.get("api_key") or "").strip() or None

    if not video_id:
        return jsonify({"error": "Missing video_id."}), 400

    try:
        result = generate_video_summary(
            video_id=video_id,
            provider=provider,
            summary_style=summary_style,
            custom_api_key=api_key,
        )
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RateLimitException as exc:
        return jsonify({"error": str(exc)}), 429
    except Exception as exc:
        logging.exception("Unexpected error in /api/summarize")
        return jsonify({"error": f"Server error: {exc}"}), 500


@app.post("/api/chat/init")
def api_chat_init():
    """Build or load the FAISS vector index for a video (Pipeline 2)."""
    data = request.get_json(silent=True) or {}
    video_id = (data.get("video_id") or "").strip()

    if not video_id:
        return jsonify({"error": "Missing video_id."}), 400

    try:
        result = build_or_load_vector_index(video_id)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logging.exception("Unexpected error in /api/chat/init")
        return jsonify({"error": f"Server error: {exc}"}), 500


@app.post("/api/chat/ask")
def api_chat_ask():
    """Answer a grounded RAG question about the loaded video (Pipeline 2)."""
    data = request.get_json(silent=True) or {}
    video_id = (data.get("video_id") or "").strip()
    question = (data.get("question") or "").strip()
    provider = (data.get("provider") or "gemini").strip().lower()
    api_key = (data.get("api_key") or "").strip() or None

    if not video_id:
        return jsonify({"error": "Missing video_id."}), 400
    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    try:
        result = ask_video_question(
            video_id=video_id,
            question=question,
            provider=provider,
            custom_api_key=api_key,
        )
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RateLimitException as exc:
        return jsonify({"error": str(exc)}), 429
    except Exception as exc:
        logging.exception("Unexpected error in /api/chat/ask")
        return jsonify({"error": f"Server error: {exc}"}), 500


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
