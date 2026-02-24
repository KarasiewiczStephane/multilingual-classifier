"""Tests for the FastAPI classification endpoints."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.utils.config import reset_config_cache

MOCK_CLASSIFICATION = {
    "primary_intent": "billing",
    "primary_confidence": 0.85,
    "secondary_intent": "complaint",
    "secondary_confidence": 0.12,
    "all_scores": {"billing": 0.85, "complaint": 0.12},
    "needs_human_review": False,
}


@pytest.fixture(autouse=True)
def _clear_cache():
    reset_config_cache()
    yield
    reset_config_cache()


@pytest.fixture(autouse=True)
def _reset_globals():
    """Reset module-level globals between tests."""
    import src.api.app as app_module

    app_module._classifier = None
    app_module._urgency_scorer = None
    app_module._lang_detector = None
    app_module._template_engine = None
    app_module._preprocessor = None
    app_module._metrics_db = None
    app_module._model_loaded = False
    yield


@pytest.fixture()
def mock_classifier():
    """Mock the ZeroShotClassifier to avoid model downloads."""
    mock_cls = MagicMock()
    mock_cls.classify.return_value = MOCK_CLASSIFICATION
    with patch("src.api.app._get_classifier", return_value=mock_cls):
        yield mock_cls


@pytest.fixture()
def client(mock_classifier) -> TestClient:
    """Create a test client with mocked classifier."""
    from src.api.app import app

    return TestClient(app)


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_check(self, client: TestClient) -> None:
        """Health endpoint should return 200."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_health_model_status(self, client: TestClient) -> None:
        """Health should report model loading status."""
        response = client.get("/health")
        data = response.json()
        assert "model_loaded" in data


class TestLanguagesEndpoint:
    """Tests for the /languages endpoint."""

    def test_list_languages(self, client: TestClient) -> None:
        """Should return list of supported languages."""
        response = client.get("/languages")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_language_format(self, client: TestClient) -> None:
        """Each language entry should have code, name, supported."""
        response = client.get("/languages")
        data = response.json()
        for lang in data:
            assert "code" in lang
            assert "name" in lang
            assert "supported" in lang


class TestClassifyEndpoint:
    """Tests for the /classify endpoint."""

    def test_classify_single(self, client: TestClient) -> None:
        """Should classify a single ticket."""
        response = client.post(
            "/classify",
            json={"text": "I was charged twice for my subscription"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "intent" in data
        assert "urgency" in data
        assert "language" in data
        assert "processing_time_ms" in data

    def test_classify_with_context(self, client: TestClient) -> None:
        """Should accept customer_name and ticket_id."""
        response = client.post(
            "/classify",
            json={
                "text": "I need help with billing",
                "customer_name": "John Doe",
                "ticket_id": "TICKET-123",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["intent"]["primary_intent"] == "billing"

    def test_classify_empty_text_rejected(self, client: TestClient) -> None:
        """Empty text should be rejected."""
        response = client.post("/classify", json={"text": ""})
        assert response.status_code == 422

    def test_classify_response_has_urgency(self, client: TestClient) -> None:
        """Response should include urgency level and score."""
        response = client.post(
            "/classify",
            json={"text": "URGENT: my system is down"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["urgency"] in ["low", "medium", "high", "critical"]
        assert isinstance(data["urgency_score"], float)

    def test_classify_response_has_language(self, client: TestClient) -> None:
        """Response should include detected language."""
        response = client.post(
            "/classify",
            json={"text": "I need help with my account please"},
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["language"], str)
        assert isinstance(data["language_confidence"], float)

    def test_classify_escalation_flag(self, client: TestClient) -> None:
        """Response should include should_escalate flag."""
        response = client.post(
            "/classify",
            json={"text": "This is an emergency"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "should_escalate" in data


class TestBatchEndpoint:
    """Tests for the /classify/batch endpoint."""

    def test_batch_classify(self, client: TestClient) -> None:
        """Should classify multiple tickets."""
        response = client.post(
            "/classify/batch",
            json={
                "tickets": [
                    {"text": "Billing issue"},
                    {"text": "Technical problem"},
                ]
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_processed"] == 2
        assert len(data["results"]) == 2
        assert "total_time_ms" in data

    def test_batch_single_item(self, client: TestClient) -> None:
        """Batch with single item should work."""
        response = client.post(
            "/classify/batch",
            json={"tickets": [{"text": "Single ticket"}]},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_processed"] == 1

    def test_batch_empty_rejected(self, client: TestClient) -> None:
        """Empty batch should be rejected."""
        response = client.post("/classify/batch", json={"tickets": []})
        assert response.status_code == 422


class TestMetricsEndpoint:
    """Tests for the /metrics endpoint."""

    def test_metrics_empty_db(self, client: TestClient, tmp_path) -> None:
        """Metrics on empty DB should return valid response."""
        import src.api.app as app_module
        from src.utils.database import MetricsDatabase

        app_module._metrics_db = MetricsDatabase(
            db_path=str(tmp_path / "test_metrics.db")
        )
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert data["total_classifications"] == 0
        assert data["per_language"] == []
        assert data["per_intent"] == []
        assert "latency" in data
        assert "model_info" in data

    def test_metrics_with_data(self, client: TestClient, tmp_path) -> None:
        """Metrics with data should return populated stats."""
        import src.api.app as app_module
        from src.utils.database import MetricsDatabase

        db = MetricsDatabase(db_path=str(tmp_path / "test_metrics.db"))
        db.log_classification(
            text="test",
            language="en",
            primary_intent="billing",
            confidence=0.85,
            urgency="low",
            was_escalated=False,
            needs_review=False,
            processing_time_ms=42.0,
        )
        app_module._metrics_db = db

        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert data["total_classifications"] == 1
        assert len(data["per_language"]) == 1
        assert data["per_language"][0]["language"] == "en"

    def test_metrics_has_model_info(self, client: TestClient, tmp_path) -> None:
        """Metrics should include model info."""
        import src.api.app as app_module
        from src.utils.database import MetricsDatabase

        app_module._metrics_db = MetricsDatabase(
            db_path=str(tmp_path / "test_metrics.db")
        )
        response = client.get("/metrics")
        data = response.json()
        assert "model_name" in data["model_info"]
        assert "intent_categories" in data["model_info"]
