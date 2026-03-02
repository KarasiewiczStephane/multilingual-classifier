"""Streamlit dashboard for the multilingual classifier.

Provides a classification demo interface with text input, intent prediction
with confidence visualization, language detection, per-language accuracy
heatmap, and urgency scoring indicators using synthetic results.
"""

import random

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Multilingual Classifier Dashboard",
    page_icon="🌐",
    layout="wide",
)

INTENT_CATEGORIES = [
    "billing",
    "technical_support",
    "account",
    "general_inquiry",
    "complaint",
    "feedback",
]

SUPPORTED_LANGUAGES = ["en", "es", "fr", "de", "pt"]

LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
}

SAMPLE_TEXTS = {
    "en": "My billing statement shows an incorrect charge from last month. Please help!",
    "es": "Mi factura muestra un cargo incorrecto del mes pasado. Necesito ayuda urgente.",
    "fr": "Ma facture montre un montant incorrect du mois dernier. Aidez-moi s'il vous plait.",
    "de": "Meine Rechnung zeigt eine falsche Abbuchung vom letzten Monat. Bitte helfen Sie mir.",
    "pt": "Minha fatura mostra uma cobranca incorreta do mes passado. Preciso de ajuda urgente.",
}


@st.cache_data
def generate_classification_result(text: str) -> dict:
    """Generate a synthetic classification result for the given text.

    Produces deterministic results based on text content hash for consistency.
    """
    rng = random.Random(hash(text) % 2**32)

    scores = {}
    remaining = 1.0
    shuffled_intents = INTENT_CATEGORIES.copy()
    rng.shuffle(shuffled_intents)

    for i, intent in enumerate(shuffled_intents):
        if i == len(shuffled_intents) - 1:
            scores[intent] = round(remaining, 4)
        else:
            score = rng.uniform(0.01, remaining * 0.6)
            scores[intent] = round(score, 4)
            remaining -= score

    sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    primary_intent, primary_conf = sorted_intents[0]
    secondary_intent, secondary_conf = sorted_intents[1]

    text_lower = text.lower()
    detected_lang = "en"
    lang_conf = 0.92
    for lang, sample in SAMPLE_TEXTS.items():
        overlap = len(set(text_lower.split()) & set(sample.lower().split()))
        if overlap > 3 and lang != "en":
            detected_lang = lang
            lang_conf = 0.85 + rng.uniform(0, 0.12)
            break

    urgency_keywords = {
        "urgent",
        "emergency",
        "broken",
        "help",
        "urgente",
        "ayuda",
        "dringend",
    }
    has_urgency = any(kw in text_lower for kw in urgency_keywords)
    urgency_score = rng.uniform(0.65, 0.95) if has_urgency else rng.uniform(0.15, 0.45)

    if urgency_score >= 0.9:
        urgency_level = "critical"
    elif urgency_score >= 0.65:
        urgency_level = "high"
    elif urgency_score >= 0.35:
        urgency_level = "medium"
    else:
        urgency_level = "low"

    return {
        "primary_intent": primary_intent,
        "primary_confidence": round(primary_conf, 4),
        "secondary_intent": secondary_intent,
        "secondary_confidence": round(secondary_conf, 4),
        "all_scores": dict(sorted_intents),
        "detected_language": detected_lang,
        "language_confidence": round(lang_conf, 4),
        "language_name": LANGUAGE_NAMES.get(detected_lang, detected_lang),
        "urgency_score": round(urgency_score, 4),
        "urgency_level": urgency_level,
        "needs_human_review": primary_conf < 0.7,
        "processing_time_ms": round(rng.uniform(15, 85), 1),
    }


@st.cache_data
def generate_language_accuracy_data() -> pd.DataFrame:
    """Generate synthetic per-language, per-intent accuracy data for heatmap."""
    rng = np.random.default_rng(42)
    rows = []
    for lang in SUPPORTED_LANGUAGES:
        for intent in INTENT_CATEGORIES:
            base_accuracy = 0.85 + rng.uniform(-0.1, 0.1)
            lang_penalty = 0.0 if lang == "en" else rng.uniform(0.0, 0.08)
            accuracy = max(0.65, min(0.98, base_accuracy - lang_penalty))
            rows.append(
                {
                    "language": LANGUAGE_NAMES[lang],
                    "intent": intent,
                    "accuracy": round(accuracy, 3),
                }
            )
    return pd.DataFrame(rows)


def render_header() -> None:
    """Render the dashboard header."""
    st.title("Multilingual Classifier Dashboard")
    st.caption("Intent classification, language detection, and urgency scoring demo")


def render_classification_demo() -> dict | None:
    """Render the text input area and return classification results."""
    st.subheader("Classification Demo")

    col1, col2 = st.columns([3, 1])
    with col1:
        text_input = st.text_area(
            "Enter text to classify:",
            value=SAMPLE_TEXTS["en"],
            height=120,
            help="Enter a support ticket or customer message in any supported language.",
        )
    with col2:
        st.markdown("**Quick examples:**")
        for lang, sample in SAMPLE_TEXTS.items():
            if st.button(f"{LANGUAGE_NAMES[lang]}", key=f"sample_{lang}"):
                st.session_state["demo_text"] = sample

    if "demo_text" in st.session_state:
        text_input = st.session_state.pop("demo_text")
        st.rerun()

    if text_input and text_input.strip():
        return generate_classification_result(text_input)
    return None


def render_intent_results(result: dict) -> None:
    """Render predicted intent with confidence bar chart."""
    st.subheader("Predicted Intent")

    col1, col2, col3 = st.columns(3)
    col1.metric("Primary Intent", result["primary_intent"].replace("_", " ").title())
    col2.metric("Confidence", f"{result['primary_confidence']:.1%}")
    col3.metric("Processing Time", f"{result['processing_time_ms']:.1f}ms")

    if result["needs_human_review"]:
        st.warning("Low confidence - flagged for human review")

    scores = result["all_scores"]
    fig = go.Figure(
        go.Bar(
            x=list(scores.values()),
            y=[k.replace("_", " ").title() for k in scores.keys()],
            orientation="h",
            marker_color=[
                "#2196F3" if k == result["primary_intent"] else "#E0E0E0"
                for k in scores.keys()
            ],
            text=[f"{v:.1%}" for v in scores.values()],
            textposition="auto",
        )
    )
    fig.update_layout(
        xaxis_title="Confidence Score",
        xaxis={"range": [0, 1]},
        height=300,
        margin={"l": 120, "r": 20, "t": 20, "b": 40},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_language_detection(result: dict) -> None:
    """Render language detection display."""
    st.subheader("Language Detection")

    col1, col2, col3 = st.columns(3)
    col1.metric("Detected Language", result["language_name"])
    col2.metric("Language Code", result["detected_language"])
    col3.metric("Confidence", f"{result['language_confidence']:.1%}")


def render_urgency_indicator(result: dict) -> None:
    """Render urgency scoring indicator."""
    st.subheader("Urgency Assessment")

    level = result["urgency_level"]
    score = result["urgency_score"]

    level_colors = {
        "low": "#4CAF50",
        "medium": "#FF9800",
        "high": "#F44336",
        "critical": "#B71C1C",
    }
    color = level_colors.get(level, "#999")

    col1, col2 = st.columns(2)
    col1.metric("Urgency Level", level.upper())
    col2.metric("Urgency Score", f"{score:.2f}")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score * 100,
            title={"text": "Urgency"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 35], "color": "#E8F5E9"},
                    {"range": [35, 65], "color": "#FFF3E0"},
                    {"range": [65, 90], "color": "#FFEBEE"},
                    {"range": [90, 100], "color": "#FFCDD2"},
                ],
            },
            number={"suffix": "%"},
        )
    )
    fig.update_layout(
        height=250,
        margin={"l": 20, "r": 20, "t": 50, "b": 10},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_language_accuracy_heatmap() -> None:
    """Render per-language accuracy heatmap across intent categories."""
    st.subheader("Per-Language Accuracy Heatmap")

    df = generate_language_accuracy_data()
    pivot = df.pivot(index="language", columns="intent", values="accuracy")

    fig = px.imshow(
        pivot,
        text_auto=".2f",
        color_continuous_scale="RdYlGn",
        zmin=0.65,
        zmax=1.0,
        aspect="auto",
    )
    fig.update_layout(
        xaxis_title="Intent Category",
        yaxis_title="Language",
        height=350,
        margin={"l": 80, "r": 20, "t": 30, "b": 60},
    )
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    """Main dashboard entry point."""
    render_header()

    st.sidebar.markdown("### Dashboard Controls")
    show_demo = st.sidebar.checkbox("Show classification demo", value=True)
    show_heatmap = st.sidebar.checkbox("Show accuracy heatmap", value=True)

    if show_demo:
        result = render_classification_demo()

        if result:
            st.markdown("---")

            col_left, col_right = st.columns([3, 2])

            with col_left:
                render_intent_results(result)

            with col_right:
                render_language_detection(result)
                render_urgency_indicator(result)

    if show_heatmap:
        st.markdown("---")
        render_language_accuracy_heatmap()


if __name__ == "__main__":
    main()
