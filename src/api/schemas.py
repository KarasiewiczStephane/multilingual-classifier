"""Pydantic request/response schemas for the classification API."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class UrgencyLevel(str, Enum):
    """Urgency level enumeration."""

    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class ClassifyRequest(BaseModel):
    """Request schema for single ticket classification."""

    text: str = Field(..., min_length=1, max_length=5000)
    customer_name: Optional[str] = None
    ticket_id: Optional[str] = None


class ClassifyBatchRequest(BaseModel):
    """Request schema for batch ticket classification."""

    tickets: list[ClassifyRequest] = Field(..., min_length=1, max_length=100)


class IntentResult(BaseModel):
    """Intent classification result."""

    primary_intent: str
    primary_confidence: float
    secondary_intent: Optional[str] = None
    secondary_confidence: Optional[float] = None
    needs_human_review: bool


class ClassifyResponse(BaseModel):
    """Response schema for single ticket classification."""

    intent: IntentResult
    urgency: UrgencyLevel
    urgency_score: float
    language: str
    language_confidence: float
    response_suggestion: Optional[dict] = None
    should_escalate: bool
    processing_time_ms: float


class BatchClassifyResponse(BaseModel):
    """Response schema for batch classification."""

    results: list[ClassifyResponse]
    total_processed: int
    total_time_ms: float


class LanguageInfo(BaseModel):
    """Language information."""

    code: str
    name: str
    supported: bool


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    version: str


class LanguageMetrics(BaseModel):
    """Per-language classification statistics."""

    language: str
    total_classifications: int
    avg_confidence: float
    review_rate: float
    escalation_rate: float


class LatencyMetrics(BaseModel):
    """Processing latency statistics."""

    mean_ms: float
    min_ms: float
    max_ms: float
    p50_ms: Optional[float] = None
    p95_ms: Optional[float] = None
    p99_ms: Optional[float] = None


class IntentMetrics(BaseModel):
    """Per-intent classification statistics."""

    intent: str
    count: int
    avg_confidence: float


class MetricsResponse(BaseModel):
    """Full metrics response including all statistics."""

    timestamp: str
    total_classifications: int
    per_language: list[LanguageMetrics]
    per_intent: list[IntentMetrics]
    latency: LatencyMetrics
    model_info: dict
