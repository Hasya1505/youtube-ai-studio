"""
LLM Factory – Unified Gemini & Groq model loader with 429 retry/backoff wrapper.
"""

import os
import time
import logging

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

load_dotenv()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom Exception
# ---------------------------------------------------------------------------

class RateLimitException(Exception):
    """Raised when the Free-Tier API rate limit is exhausted after retries."""
    pass


# ---------------------------------------------------------------------------
# Model Loader
# ---------------------------------------------------------------------------

def get_chat_model(provider: str, custom_api_key: str | None = None, fast_mode: bool = False):
    """
    Return a LangChain ChatModel for the requested provider.

    Parameters
    ----------
    provider : "gemini" | "groq"
    custom_api_key : Optional key that overrides the .env value.
    fast_mode : If True and provider == "groq", use the smaller 8B model.
    """
    provider = provider.strip().lower()

    if provider == "gemini":
        api_key = (custom_api_key or "").strip() or os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "Gemini API key is missing. Add GEMINI_API_KEY in .env or enter it in the UI."
            )
        return ChatGoogleGenerativeAI(
            model="gemini-3.8-flash-lite",
            google_api_key=api_key,
            temperature=0.2,
            max_output_tokens=1500,
        )

    if provider == "groq":
        api_key = (custom_api_key or "").strip() or os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError(
                "Groq API key is missing. Add GROQ_API_KEY in .env or enter it in the UI."
            )
        model_name = "openai/gpt-oss-20b"
        return ChatGroq(
            model=model_name,
            groq_api_key=api_key,
            temperature=0.2,
            max_tokens=1500,
        )

    raise ValueError(f"Unknown provider '{provider}'. Choose 'gemini' or 'groq'.")


# ---------------------------------------------------------------------------
# Rate-Limit-Aware Invoker
# ---------------------------------------------------------------------------

_RATE_LIMIT_SIGNALS = ("429", "resource_exhausted", "rate_limit", "rate limit")


def invoke_with_retry(chain, payload: dict, max_retries: int = 2) -> str:
    """
    Execute *chain.invoke(payload)* with automatic back-off on 429 errors.
    Returns the string content of the response.
    """
    last_exc = None

    for attempt in range(max_retries + 1):
        try:
            result = chain.invoke(payload)
            # LangChain ChatModel returns an AIMessage; StrOutputParser returns str
            return result if isinstance(result, str) else str(result)
        except Exception as exc:
            error_text = str(exc).lower()
            is_rate_limit = any(sig in error_text for sig in _RATE_LIMIT_SIGNALS)
            if not is_rate_limit:
                raise  # Not a rate-limit error – propagate immediately

            last_exc = exc
            wait_seconds = 3 * (attempt + 1)
            logger.warning(
                "[RATE-LIMIT] Attempt %d/%d hit 429. Waiting %ds …",
                attempt + 1,
                max_retries + 1,
                wait_seconds,
            )
            time.sleep(wait_seconds)

    raise RateLimitException(
        "Free Tier rate limit reached. Please wait 15–30 seconds and try again, "
        "or switch providers."
    ) from last_exc
