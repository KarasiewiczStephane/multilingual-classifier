"""Tests for the response template engine."""

import pytest

from src.responses.template_engine import ResponseTemplateEngine


@pytest.fixture()
def engine() -> ResponseTemplateEngine:
    """Create a ResponseTemplateEngine with default templates."""
    return ResponseTemplateEngine()


class TestTemplateLoading:
    """Tests for template loading."""

    def test_loads_supported_languages(self, engine: ResponseTemplateEngine) -> None:
        """Should discover language directories."""
        assert "en" in engine.supported_languages
        assert "es" in engine.supported_languages
        assert "fr" in engine.supported_languages

    def test_loads_english_templates(self, engine: ResponseTemplateEngine) -> None:
        """Should load English intent templates."""
        assert "billing" in engine.templates.get("en", {})
        assert "technical_support" in engine.templates.get("en", {})

    def test_loads_base_templates(self, engine: ResponseTemplateEngine) -> None:
        """Should load base fallback templates."""
        assert "fallback" in engine._base_templates

    def test_list_available_templates(self, engine: ResponseTemplateEngine) -> None:
        """Should list templates by language."""
        available = engine.list_available_templates()
        assert isinstance(available, dict)
        assert "en" in available
        assert "billing" in available["en"]


class TestGetTemplate:
    """Tests for template retrieval."""

    def test_get_english_billing_default(self, engine: ResponseTemplateEngine) -> None:
        """Should return default billing template for English."""
        template = engine.get_template("billing", "en", "low")
        assert template is not None
        assert "subject" in template
        assert "body" in template

    def test_get_english_billing_urgent(self, engine: ResponseTemplateEngine) -> None:
        """Should return high urgency billing template."""
        template = engine.get_template("billing", "en", "high")
        assert template is not None
        assert (
            "URGENT" in template.get("subject", "").upper()
            or "PRIORITY" in template.get("subject", "").upper()
        )

    def test_get_spanish_template(self, engine: ResponseTemplateEngine) -> None:
        """Should return Spanish billing template."""
        template = engine.get_template("billing", "es", "low")
        assert template is not None
        assert "subject" in template

    def test_fallback_to_english(self, engine: ResponseTemplateEngine) -> None:
        """Unknown language should fall back to English."""
        template = engine.get_template("billing", "zh", "low")
        assert template is not None

    def test_fallback_to_base(self, engine: ResponseTemplateEngine) -> None:
        """Unknown intent should fall back to base templates."""
        template = engine.get_template("nonexistent_intent", "xx", "low")
        assert template is not None
        assert "body" in template

    def test_critical_urgency_uses_high_template(
        self, engine: ResponseTemplateEngine
    ) -> None:
        """Critical urgency should use high_urgency template variant."""
        template = engine.get_template("billing", "en", "critical")
        assert template is not None


class TestRenderResponse:
    """Tests for template rendering with personalization."""

    def test_render_with_context(self, engine: ResponseTemplateEngine) -> None:
        """Should fill in personalization slots."""
        result = engine.render_response(
            intent="billing",
            language="en",
            urgency="low",
            context={"customer_name": "John Doe", "ticket_id": "12345"},
        )
        assert result is not None
        assert "John Doe" in result["body"]
        assert "12345" in result["body"]

    def test_render_default_context(self, engine: ResponseTemplateEngine) -> None:
        """Should use default values when no context provided."""
        result = engine.render_response(intent="billing", language="en")
        assert result is not None
        assert "Valued Customer" in result["body"]

    def test_render_spanish_template(self, engine: ResponseTemplateEngine) -> None:
        """Should render Spanish templates."""
        result = engine.render_response(
            intent="billing",
            language="es",
            context={"customer_name": "Juan García", "ticket_id": "99999"},
        )
        assert result is not None
        assert "Juan García" in result["body"]

    def test_render_nonexistent_returns_fallback(
        self, engine: ResponseTemplateEngine
    ) -> None:
        """Should return fallback for unknown intents."""
        result = engine.render_response(intent="unknown_intent", language="en")
        assert result is not None

    def test_render_high_urgency(self, engine: ResponseTemplateEngine) -> None:
        """High urgency should select urgent template."""
        result = engine.render_response(
            intent="billing",
            language="en",
            urgency="high",
            context={"customer_name": "Jane", "ticket_id": "5555"},
        )
        assert result is not None
        assert "5555" in result["body"]


class TestEdgeCases:
    """Tests for edge cases in template engine."""

    def test_empty_templates_dir(self, tmp_path) -> None:
        """Should handle empty templates directory."""
        engine = ResponseTemplateEngine(templates_dir=str(tmp_path))
        assert len(engine.supported_languages) == 0

    def test_missing_templates_dir(self, tmp_path) -> None:
        """Should handle missing templates directory."""
        engine = ResponseTemplateEngine(templates_dir=str(tmp_path / "nonexistent"))
        assert len(engine.templates) == 0
