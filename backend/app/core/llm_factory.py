"""
GCIS Backend — LLM Factory
Provider-agnostic LLM instantiation.
All LLM-backed agents call get_llm() — zero code changes to swap providers.

Supported providers:
  groq   → Groq + Llama 3.3 70B Versatile  (default, free tier)
  gemini → Google Gemini 2.5 Flash          (fallback)
  ollama → Ollama local model               (offline demo fallback)
"""
import structlog
from langchain_core.language_models.chat_models import BaseChatModel

from app.core.config import settings

log = structlog.get_logger(__name__)


def get_llm(temperature: float = 0.1) -> BaseChatModel:
    """
    Return a LangChain-compatible chat model based on LLM_PROVIDER setting.

    Args:
        temperature: Sampling temperature. Default 0.1 for factual, constrained narration.

    Returns:
        BaseChatModel instance ready for LangChain/LangGraph use.

    Raises:
        ValueError: If an unsupported LLM_PROVIDER is configured.
    """
    provider = settings.LLM_PROVIDER.lower()
    model = settings.LLM_MODEL

    log.info("llm_factory.instantiating", provider=provider, model=model)

    if provider == "groq":
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(
                model=model,
                api_key=settings.GROQ_API_KEY,
                temperature=temperature,
                max_retries=3,
            )
        except ImportError:
            raise ImportError("langchain-groq not installed. Run: pip install langchain-groq")

    elif provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=model,
                google_api_key=settings.GOOGLE_API_KEY,
                temperature=temperature,
            )
        except ImportError:
            raise ImportError(
                "langchain-google-genai not installed. Run: pip install langchain-google-genai"
            )

    elif provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=model,
                temperature=temperature,
            )
        except ImportError:
            raise ImportError(
                "langchain-ollama not installed. Run: pip install langchain-ollama"
            )

    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER: '{provider}'. "
            "Choose from: groq, gemini, ollama"
        )


def get_llm_with_fallback(temperature: float = 0.1) -> BaseChatModel:
    """
    Attempt primary provider (Groq). Fall back to Gemini, then Ollama.
    Useful for demo resilience during live hackathon presentation.
    """
    fallback_chain = [
        ("groq",   "llama-3.3-70b-versatile"),
        ("gemini", "gemini-2.5-flash"),
        ("ollama", "llama3.1:8b"),
    ]

    for provider, model in fallback_chain:
        try:
            original_provider = settings.LLM_PROVIDER
            original_model = settings.LLM_MODEL
            # Temporarily override for factory call
            settings.LLM_PROVIDER = provider
            settings.LLM_MODEL = model
            llm = get_llm(temperature)
            settings.LLM_PROVIDER = original_provider
            settings.LLM_MODEL = original_model
            log.info("llm_factory.using_provider", provider=provider, model=model)
            return llm
        except Exception as e:
            log.warning("llm_factory.provider_failed", provider=provider, error=str(e))
            continue

    raise RuntimeError("All LLM providers failed. Check API keys and connectivity.")
