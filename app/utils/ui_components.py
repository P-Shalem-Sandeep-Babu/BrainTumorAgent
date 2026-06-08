"""
UI Components for the Brain Tumor Agent Streamlit App.

Provides:
- Medical-style CSS theming
- Confidence visualization charts
- Result cards for predictions
- Severity indicators
- Recommendation cards
- Prediction history management

Usage:
    from app.utils.ui_components import apply_custom_css, render_result_card
"""

import streamlit as st
import plotly.graph_objects as go
from datetime import datetime


# =============================================================================
# COLOR PALETTE (Medical/Clinical Theme)
# =============================================================================

COLORS = {
    "primary": "#0077B6",       # Deep medical blue
    "primary_light": "#00B4D8", # Light blue accent
    "success": "#2DC653",       # Green (no tumor / good)
    "warning": "#F4A261",       # Orange (moderate risk)
    "danger": "#E63946",        # Red (high risk / emergency)
    "info": "#457B9D",          # Muted blue (info cards)
    "bg_card": "#F8F9FA",       # Light card background
    "text_dark": "#1D3557",     # Dark text
    "text_muted": "#6C757D",    # Muted text
    "border": "#DEE2E6",        # Light border
}

# Severity color mapping
SEVERITY_COLORS = {
    0: COLORS["success"],   # No tumor - green
    1: "#457B9D",           # Pituitary - blue
    2: COLORS["warning"],   # Meningioma - orange
    3: COLORS["danger"],    # Glioma - red
}

SEVERITY_LABELS = {
    0: "No Concern",
    1: "Low",
    2: "Moderate",
    3: "High",
}


# =============================================================================
# CUSTOM CSS
# =============================================================================

def apply_custom_css():
    """
    Apply medical-style CSS theme to the Streamlit app.

    This overrides default Streamlit styles to give a clean,
    clinical look suitable for a medical application.
    """
    st.markdown("""
    <style>
        /* ---- Global font and spacing ---- */
        .stApp {
            font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif;
        }

        /* ---- Main header ---- */
        .main-header {
            background: linear-gradient(135deg, #0077B6 0%, #00B4D8 100%);
            padding: 1.5rem 2rem;
            border-radius: 12px;
            color: white;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 15px rgba(0, 119, 182, 0.2);
        }
        .main-header h1 {
            color: white !important;
            font-size: 1.8rem;
            margin: 0;
            font-weight: 700;
        }
        .main-header p {
            color: rgba(255,255,255,0.9);
            margin: 0.3rem 0 0 0;
            font-size: 0.95rem;
        }

        /* ---- Result cards ---- */
        .result-card {
            background: white;
            border: 1px solid #DEE2E6;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            transition: box-shadow 0.2s ease;
        }
        .result-card:hover {
            box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        }

        /* ---- Tumor type badge ---- */
        .tumor-badge {
            display: inline-block;
            padding: 0.4rem 1.2rem;
            border-radius: 20px;
            font-weight: 600;
            font-size: 1.1rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .tumor-badge.glioma {
            background: rgba(230, 57, 70, 0.1);
            color: #E63946;
            border: 2px solid #E63946;
        }
        .tumor-badge.meningioma {
            background: rgba(244, 162, 97, 0.1);
            color: #E76F51;
            border: 2px solid #F4A261;
        }
        .tumor-badge.pituitary {
            background: rgba(69, 123, 157, 0.1);
            color: #457B9D;
            border: 2px solid #457B9D;
        }
        .tumor-badge.notumor {
            background: rgba(45, 198, 83, 0.1);
            color: #2D8A4E;
            border: 2px solid #2DC653;
        }

        /* ---- Confidence meter ---- */
        .confidence-meter {
            background: #E9ECEF;
            border-radius: 10px;
            height: 12px;
            overflow: hidden;
            margin: 0.5rem 0;
        }
        .confidence-fill {
            height: 100%;
            border-radius: 10px;
            transition: width 0.8s ease;
        }

        /* ---- Severity indicator ---- */
        .severity-indicator {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.6rem 1rem;
            border-radius: 8px;
            font-weight: 500;
            margin: 0.5rem 0;
        }

        /* ---- Recommendation card ---- */
        .recommendation-card {
            background: linear-gradient(135deg, #F8F9FA 0%, #E9ECEF 100%);
            border-left: 4px solid #0077B6;
            border-radius: 0 12px 12px 0;
            padding: 1.2rem 1.5rem;
            margin: 1rem 0;
        }
        .recommendation-card .action-label {
            font-size: 0.85rem;
            color: #6C757D;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 0.3rem;
        }
        .recommendation-card .action-text {
            font-size: 1.2rem;
            font-weight: 600;
            color: #1D3557;
        }

        /* ---- Stat boxes ---- */
        .stat-box {
            text-align: center;
            padding: 1rem;
            border-radius: 10px;
            background: white;
            border: 1px solid #DEE2E6;
        }
        .stat-box .stat-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #0077B6;
        }
        .stat-box .stat-label {
            font-size: 0.8rem;
            color: #6C757D;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* ---- History table ---- */
        .history-item {
            display: flex;
            align-items: center;
            padding: 0.8rem 1rem;
            border-bottom: 1px solid #E9ECEF;
            gap: 1rem;
        }
        .history-item:last-child {
            border-bottom: none;
        }

        /* ---- Disclaimer ---- */
        .disclaimer {
            background: #FFF3CD;
            border: 1px solid #FFEAA7;
            border-radius: 8px;
            padding: 0.8rem 1.2rem;
            font-size: 0.85rem;
            color: #856404;
            margin-top: 1.5rem;
        }

        /* ---- Loading animation ---- */
        @keyframes pulse {
            0%, 100% { opacity: 0.4; }
            50% { opacity: 1; }
        }
        .loading-pulse {
            animation: pulse 1.5s ease-in-out infinite;
        }
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# HEADER COMPONENT
# =============================================================================

def render_header(title: str, subtitle: str = ""):
    """
    Render a styled page header with gradient background.

    Args:
        title: Main heading text.
        subtitle: Optional subheading text.
    """
    subtitle_html = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(f"""
    <div class="main-header">
        <h1>{title}</h1>
        {subtitle_html}
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# CONFIDENCE CHART (Plotly)
# =============================================================================

def render_confidence_chart(probabilities: dict, predicted_class: str) -> go.Figure:
    """
    Create a horizontal bar chart showing confidence for each tumor class.

    Args:
        probabilities: Dict of {class_name: probability}.
        predicted_class: The predicted class name (highlighted).

    Returns:
        Plotly Figure object.
    """
    class_names = list(probabilities.keys())
    probs = list(probabilities.values())

    # Color bars: predicted class gets its severity color, others are muted
    colors = []
    for name in class_names:
        if name == predicted_class:
            severity = {"glioma": 3, "meningioma": 2, "pituitary": 1, "notumor": 0}
            colors.append(SEVERITY_COLORS.get(severity.get(name, 0), COLORS["primary"]))
        else:
            colors.append("#BDD5EA")

    # Create horizontal bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=class_names,
        x=[p * 100 for p in probs],
        orientation='h',
        marker_color=colors,
        text=[f"{p*100:.1f}%" for p in probs],
        textposition='outside',
        textfont=dict(size=14, color=COLORS["text_dark"]),
        hovertemplate="<b>%{y}</b><br>Confidence: %{x:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        xaxis=dict(
            title="Confidence (%)",
            range=[0, 105],
            gridcolor="#E9ECEF",
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=13, color=COLORS["text_dark"]),
            autorange="reversed",  # Keep order top-to-bottom
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=30, t=10, b=40),
        height=220,
        showlegend=False,
    )

    return fig


# =============================================================================
# PROBABILITY DONUT CHART
# =============================================================================

def render_donut_chart(probabilities: dict) -> go.Figure:
    """
    Create a donut chart showing probability distribution.

    Args:
        probabilities: Dict of {class_name: probability}.

    Returns:
        Plotly Figure object.
    """
    class_names = list(probabilities.keys())
    probs = [p * 100 for p in probabilities.values()]

    severity_map = {"glioma": 3, "meningioma": 2, "pituitary": 1, "notumor": 0}
    colors = [SEVERITY_COLORS.get(severity_map.get(n, 0), COLORS["info"]) for n in class_names]

    fig = go.Figure(go.Pie(
        labels=class_names,
        values=probs,
        hole=0.55,
        marker_colors=colors,
        textinfo='label+percent',
        textposition='outside',
        textfont=dict(size=12),
        hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        showlegend=False,
        margin=dict(l=20, r=20, t=10, b=10),
        height=280,
        paper_bgcolor="white",
        annotations=[dict(
            text="Probability",
            x=0.5, y=0.5,
            font_size=13,
            font_color=COLORS["text_muted"],
            showarrow=False,
        )],
    )

    return fig


# =============================================================================
# RESULT CARDS
# =============================================================================

def render_tumor_result(tumor_type: str, confidence: float, severity: int, source: str = "Trained ViT"):
    """
    Render the main tumor classification result card.

    Args:
        tumor_type: Predicted tumor class name.
        confidence: Prediction confidence (0.0-1.0).
        severity: Severity level (0-3).
        source: Where the prediction came from (e.g., "Trained ViT" or "Demo").
    """
    sev_color = SEVERITY_COLORS.get(severity, COLORS["info"])
    sev_label = SEVERITY_LABELS.get(severity, "Unknown")

    # Confidence bar color
    if confidence >= 0.8:
        conf_color = COLORS["success"]
    elif confidence >= 0.5:
        conf_color = COLORS["warning"]
    else:
        conf_color = COLORS["danger"]

    # Source badge styling
    is_demo = source.lower() in ("demo", "simulated")
    source_bg = "#FFF3CD" if is_demo else "#D1ECF1"
    source_color = "#856404" if is_demo else "#0C5460"
    source_border = "#FFEAA7" if is_demo else "#BEE5EB"
    source_icon = "&#9888;" if is_demo else "&#10003;"  # warning checkmark vs checkmark

    st.markdown(f"""
    <div class="result-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
            <span class="tumor-badge {tumor_type}">{tumor_type}</span>
            <span style="
                background: {source_bg};
                color: {source_color};
                border: 1px solid {source_border};
                padding: 0.25rem 0.7rem;
                border-radius: 12px;
                font-size: 0.75rem;
                font-weight: 600;
            ">{source_icon} {source}</span>
        </div>
        <div style="margin-bottom: 0.5rem;">
            <span style="color: {COLORS['text_muted']}; font-size: 0.9rem;">Confidence</span>
            <span style="float: right; font-weight: 600; color: {COLORS['text_dark']};">
                {confidence:.1%}
            </span>
        </div>
        <div class="confidence-meter">
            <div class="confidence-fill" style="
                width: {confidence * 100}%;
                background: linear-gradient(90deg, {conf_color} 0%, {conf_color}CC 100%);
            "></div>
        </div>
        <div class="severity-indicator" style="background: {sev_color}15; border: 1px solid {sev_color}40;">
            <span style="color: {sev_color}; font-weight: 600;">Severity: {sev_label}</span>
            <span style="color: {COLORS['text_muted']}; font-size: 0.85rem;">
                (Level {severity}/3)
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_recommendation_card(recommendation: str, action_index: int, source: str = "DQN Agent"):
    """
    Render the treatment recommendation card.

    Args:
        recommendation: Recommended treatment action text.
        action_index: Action index (0-3) for color coding.
        source: Where the recommendation came from (e.g., "DQN Agent" or "Rule-based").
    """
    # Color based on urgency
    urgency_colors = {
        0: COLORS["success"],   # Monitor
        1: COLORS["info"],      # Specialist
        2: COLORS["warning"],   # Biopsy
        3: COLORS["danger"],    # Emergency
    }
    color = urgency_colors.get(action_index, COLORS["primary"])

    urgency_labels = {
        0: "Routine",
        1: "Advised",
        2: "Recommended",
        3: "Urgent",
    }
    urgency = urgency_labels.get(action_index, "Standard")

    # Source badge styling
    is_fallback = source.lower() in ("rule-based", "rule-based fallback", "demo")
    source_bg = "#FFF3CD" if is_fallback else "#D1ECF1"
    source_color = "#856404" if is_fallback else "#0C5460"
    source_border = "#FFEAA7" if is_fallback else "#BEE5EB"
    source_icon = "&#9881;" if is_fallback else "&#10003;"  # gear vs checkmark

    st.markdown(f"""
    <div class="recommendation-card" style="border-left-color: {color};">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div class="action-label">RL Agent Recommendation</div>
            <span style="
                background: {source_bg};
                color: {source_color};
                border: 1px solid {source_border};
                padding: 0.2rem 0.6rem;
                border-radius: 12px;
                font-size: 0.7rem;
                font-weight: 600;
            ">{source_icon} {source}</span>
        </div>
        <div class="action-text">{recommendation}</div>
        <div style="margin-top: 0.5rem;">
            <span style="
                background: {color}20;
                color: {color};
                padding: 0.2rem 0.8rem;
                border-radius: 12px;
                font-size: 0.8rem;
                font-weight: 600;
            ">{urgency}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PREDICTION HISTORY
# =============================================================================

def init_history():
    """Initialize prediction history in session state."""
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []


def add_to_history(result: dict, filename: str):
    """
    Add a prediction result to the session history.

    Args:
        result: Prediction result dictionary from Predictor.predict().
        filename: Name of the uploaded file.
    """
    init_history()
    entry = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "filename": filename,
        "tumor_type": result["tumor_type"],
        "confidence": result["confidence"],
        "severity": result["severity"],
        "recommendation": result["recommendation"],
        "vit_source": result.get("vit_source", "Trained ViT"),
        "rec_source": result.get("rec_source", "DQN Agent"),
        "demo_mode": result.get("demo_mode", False),
    }
    st.session_state.prediction_history.insert(0, entry)

    # Keep only last 20 predictions
    if len(st.session_state.prediction_history) > 20:
        st.session_state.prediction_history = st.session_state.prediction_history[:20]


def render_history():
    """Render the prediction history sidebar section."""
    init_history()

    if not st.session_state.prediction_history:
        st.info("No predictions yet. Upload an MRI scan to get started.")
        return

    st.markdown(f"**{len(st.session_state.prediction_history)} prediction(s)**")

    for i, entry in enumerate(st.session_state.prediction_history):
        sev_color = SEVERITY_COLORS.get(entry["severity"], COLORS["info"])
        is_demo = entry.get("demo_mode", False)
        src_label = "Demo" if is_demo else entry.get("vit_source", "ViT")
        src_color = "#856404" if is_demo else "#0C5460"
        src_bg = "#FFF3CD" if is_demo else "#D1ECF1"
        st.markdown(f"""
        <div class="history-item">
            <div style="
                width: 10px; height: 10px;
                border-radius: 50%;
                background: {sev_color};
                flex-shrink: 0;
            "></div>
            <div style="flex-grow: 1;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 600; color: {COLORS['text_dark']};">
                        {entry['tumor_type'].capitalize()}
                    </span>
                    <span style="
                        background: {src_bg};
                        color: {src_color};
                        padding: 0.1rem 0.4rem;
                        border-radius: 8px;
                        font-size: 0.65rem;
                        font-weight: 600;
                    ">{src_label}</span>
                </div>
                <div style="font-size: 0.8rem; color: {COLORS['text_muted']};">
                    {entry['filename']} | {entry['confidence']:.1%} | {entry['timestamp']}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def clear_history():
    """Clear prediction history."""
    st.session_state.prediction_history = []


# =============================================================================
# DISCLAIMER
# =============================================================================

def render_disclaimer():
    """Render the medical disclaimer notice."""
    st.markdown("""
    <div class="disclaimer">
        <strong>Medical Disclaimer:</strong> This tool is NOT a substitute for professional medical
        diagnosis. Always consult a qualified healthcare provider for medical
        decisions.
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# LOADING ANIMATION
# =============================================================================

def render_loading_spinner(message: str = "Analyzing MRI scan..."):
    """
    Show a styled loading message during inference.

    Args:
        message: Text to display while loading.
    """
    return st.spinner(message)
