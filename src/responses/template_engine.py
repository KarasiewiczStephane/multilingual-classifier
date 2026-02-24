"""Response template engine with multi-language support.

Loads YAML-based response templates organized by language and intent,
with personalization slot filling and urgency-gated template selection.
"""

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class ResponseTemplateEngine:
    """Manages and renders response templates for support ticket replies.

    Templates are organized by language and intent category, with
    urgency-gated variants (default, high_urgency) for each.

    Args:
        templates_dir: Path to the directory containing template files.
    """

    def __init__(self, templates_dir: str = "src/responses/templates") -> None:
        self.templates_dir = Path(templates_dir)
        self.templates: dict[str, dict[str, dict]] = {}
        self.supported_languages: set[str] = set()
        self._base_templates: dict = {}
        self._load_all_templates()

    def _load_all_templates(self) -> None:
        """Load all template YAML files from the templates directory."""
        # Load base fallback templates
        base_path = self.templates_dir / "base.yaml"
        if base_path.exists():
            with open(base_path) as f:
                self._base_templates = yaml.safe_load(f) or {}
            logger.info("Loaded base fallback templates")

        if not self.templates_dir.exists():
            logger.warning("Templates directory not found: %s", self.templates_dir)
            return

        # Load per-language templates
        for lang_dir in sorted(self.templates_dir.iterdir()):
            if lang_dir.is_dir():
                lang = lang_dir.name
                self.supported_languages.add(lang)
                self.templates[lang] = {}

                for template_file in sorted(lang_dir.glob("*.yaml")):
                    intent = template_file.stem
                    with open(template_file) as f:
                        data = yaml.safe_load(f) or {}
                    self.templates[lang][intent] = data.get(intent, data)

                logger.info(
                    "Loaded %d templates for language=%s",
                    len(self.templates[lang]),
                    lang,
                )

    def get_template(
        self,
        intent: str,
        language: str = "en",
        urgency: str = "low",
    ) -> dict[str, str] | None:
        """Retrieve the appropriate response template.

        Follows a fallback chain: requested language -> English -> base.
        Selects urgency variant based on urgency level.

        Args:
            intent: The classified intent category.
            language: The detected or requested language code.
            urgency: The urgency level (low, medium, high, critical).

        Returns:
            Template dict with 'subject' and 'body' keys, or None.
        """
        template_key = "high_urgency" if urgency in ("high", "critical") else "default"

        # Try requested language first
        template = self._find_template(language, intent, template_key)
        if template:
            return template

        # Fallback to English
        if language != "en":
            template = self._find_template("en", intent, template_key)
            if template:
                return template

        # Fallback to base templates
        fallback = self._base_templates.get("fallback", {})
        return fallback.get(template_key, fallback.get("default"))

    def _find_template(
        self, language: str, intent: str, template_key: str
    ) -> dict[str, str] | None:
        """Look up a template for a specific language/intent/urgency combo.

        Args:
            language: Language code.
            intent: Intent category.
            template_key: Urgency variant key.

        Returns:
            Template dict or None if not found.
        """
        lang_templates = self.templates.get(language, {})
        intent_templates = lang_templates.get(intent, {})
        template = intent_templates.get(template_key)
        if template:
            return template
        return intent_templates.get("default")

    def render_response(
        self,
        intent: str,
        language: str = "en",
        urgency: str = "low",
        context: dict[str, Any] | None = None,
    ) -> dict[str, str] | None:
        """Render a response template with personalization context.

        Retrieves the appropriate template and fills in personalization
        slots like {customer_name} and {ticket_id}.

        Args:
            intent: The classified intent category.
            language: The detected language code.
            urgency: The urgency level.
            context: Dictionary of personalization values to fill in.

        Returns:
            Rendered template dict with filled subject and body,
            or None if no template is found.
        """
        template = self.get_template(intent, language, urgency)
        if not template:
            return None

        ctx = {
            "customer_name": "Valued Customer",
            "ticket_id": "N/A",
        }
        if context:
            ctx.update(context)

        rendered: dict[str, str] = {}
        for key in ("subject", "body"):
            value = template.get(key, "")
            try:
                rendered[key] = value.format(**ctx)
            except (KeyError, AttributeError):
                rendered[key] = value

        return rendered

    def list_available_templates(self) -> dict[str, list[str]]:
        """List all available templates organized by language.

        Returns:
            Dictionary mapping language codes to lists of intent names.
        """
        return {lang: list(intents.keys()) for lang, intents in self.templates.items()}
