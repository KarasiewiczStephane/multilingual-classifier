"""FastAPI application for multilingual ticket classification.

Provides REST endpoints for single and batch classification,
language listing, health checks, and metrics retrieval.
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    BatchClassifyResponse,
    ClassifyBatchRequest,
    ClassifyRequest,
    ClassifyResponse,
    HealthResponse,
    IntentResult,
    LanguageInfo,
    UrgencyLevel,
)
from src.data.preprocessor import TextPreprocessor
from src.models.language_detector import LanguageDetector
from src.models.urgency_scorer import UrgencyScorer
from src.responses.template_engine import ResponseTemplateEngine
from src.utils.config import load_config

logger = logging.getLogger(__name__)

_classifier = None
_urgency_scorer = None
_lang_detector = None
_template_engine = None
_preprocessor = None
_model_loaded = False


def _get_classifier():
    """Lazy-load the zero-shot classifier to avoid import-time model loading."""
    global _classifier, _model_loaded
    if _classifier is None:
        from src.models.zero_shot_classifier import ZeroShotClassifier

        config = load_config()
        logger.info("Loading zero-shot classifier...")
        _classifier = ZeroShotClassifier(
            model_name=config.model.zero_shot_model,
            candidate_labels=config.classification.intent_categories,
        )
        _model_loaded = True
    return _classifier


def _get_urgency_scorer() -> UrgencyScorer:
    """Get or create the urgency scorer instance."""
    global _urgency_scorer
    if _urgency_scorer is None:
        _urgency_scorer = UrgencyScorer()
    return _urgency_scorer


def _get_lang_detector() -> LanguageDetector:
    """Get or create the language detector instance."""
    global _lang_detector
    if _lang_detector is None:
        _lang_detector = LanguageDetector()
    return _lang_detector


def _get_template_engine() -> ResponseTemplateEngine:
    """Get or create the template engine instance."""
    global _template_engine
    if _template_engine is None:
        _template_engine = ResponseTemplateEngine()
    return _template_engine


def _get_preprocessor() -> TextPreprocessor:
    """Get or create the text preprocessor instance."""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = TextPreprocessor()
    return _preprocessor


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup/shutdown."""
    logger.info("Starting multilingual classifier API")
    yield
    logger.info("Shutting down multilingual classifier API")


app = FastAPI(
    title="Multilingual Customer Support Classifier",
    description="Zero-shot classification of customer support tickets with language detection and urgency scoring",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _classify_single(
    text: str, customer_name: str | None, ticket_id: str | None
) -> dict:
    """Core classification logic for a single ticket.

    Args:
        text: Ticket text to classify.
        customer_name: Optional customer name for response templates.
        ticket_id: Optional ticket ID for response templates.

    Returns:
        Dictionary with all classification results.
    """
    preprocessor = _get_preprocessor()
    lang_detector = _get_lang_detector()
    classifier = _get_classifier()
    urgency_scorer = _get_urgency_scorer()
    template_engine = _get_template_engine()

    cleaned = preprocessor.clean_text(text)
    lang, lang_conf = lang_detector.detect_language(cleaned, return_confidence=True)
    classification = classifier.classify(cleaned)
    urgency_result = urgency_scorer.score(
        cleaned,
        language=lang,
        classification_confidence=classification["primary_confidence"],
    )

    context = {
        "customer_name": customer_name or "Valued Customer",
        "ticket_id": ticket_id or "N/A",
    }
    response_suggestion = template_engine.render_response(
        intent=classification["primary_intent"],
        language=lang,
        urgency=urgency_result.level,
        context=context,
    )

    return {
        "intent": IntentResult(
            primary_intent=classification["primary_intent"],
            primary_confidence=classification["primary_confidence"],
            secondary_intent=classification["secondary_intent"],
            secondary_confidence=classification["secondary_confidence"],
            needs_human_review=classification["needs_human_review"],
        ),
        "urgency": UrgencyLevel(urgency_result.level),
        "urgency_score": urgency_result.score,
        "language": lang,
        "language_confidence": round(lang_conf, 4),
        "response_suggestion": response_suggestion,
        "should_escalate": urgency_result.should_escalate,
    }


@app.post("/classify", response_model=ClassifyResponse)
async def classify_ticket(request: ClassifyRequest) -> ClassifyResponse:
    """Classify a single customer support ticket.

    Performs language detection, intent classification, urgency scoring,
    and response template generation.
    """
    start = time.perf_counter()
    try:
        result = _classify_single(
            request.text, request.customer_name, request.ticket_id
        )
    except Exception as e:
        logger.error("Classification failed: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Classification failed: {e}")
    elapsed = (time.perf_counter() - start) * 1000
    return ClassifyResponse(**result, processing_time_ms=round(elapsed, 2))


@app.post("/classify/batch", response_model=BatchClassifyResponse)
async def classify_batch(request: ClassifyBatchRequest) -> BatchClassifyResponse:
    """Classify a batch of customer support tickets (up to 100)."""
    config = load_config()
    if len(request.tickets) > config.api.batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size exceeds maximum of {config.api.batch_size}",
        )

    start = time.perf_counter()
    results: list[ClassifyResponse] = []

    for ticket in request.tickets:
        ticket_start = time.perf_counter()
        try:
            result = _classify_single(
                ticket.text, ticket.customer_name, ticket.ticket_id
            )
            elapsed = (time.perf_counter() - ticket_start) * 1000
            results.append(
                ClassifyResponse(**result, processing_time_ms=round(elapsed, 2))
            )
        except Exception as e:
            logger.error("Batch item failed: %s", str(e))
            raise HTTPException(status_code=500, detail=f"Classification failed: {e}")

    total_elapsed = (time.perf_counter() - start) * 1000
    return BatchClassifyResponse(
        results=results,
        total_processed=len(results),
        total_time_ms=round(total_elapsed, 2),
    )


LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "it": "Italian",
    "nl": "Dutch",
    "pl": "Polish",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "tr": "Turkish",
    "sv": "Swedish",
    "da": "Danish",
}


@app.get("/languages", response_model=list[LanguageInfo])
async def list_languages() -> list[LanguageInfo]:
    """List all supported languages for classification."""
    config = load_config()
    return [
        LanguageInfo(
            code=lang,
            name=LANGUAGE_NAMES.get(lang, lang),
            supported=True,
        )
        for lang in config.languages.supported
    ]


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check API health and model loading status."""
    return HealthResponse(
        status="healthy",
        model_loaded=_model_loaded,
        version="1.0.0",
    )
