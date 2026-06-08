"""
Brain Tumor Agent - Streamlit Web Application.

Main entry point for the multi-page Streamlit app.

Usage:
    streamlit run app/main.py

Pages:
    1. Dataset Viewer    - Browse and explore MRI images
    2. Training Dashboard - Monitor RL agent training
    3. Prediction        - Upload an MRI and get a prediction
"""

import streamlit as st
from app.utils.ui_components import apply_custom_css, render_disclaimer

# -----------------------------------------------------------------------------
# Page configuration (must be first Streamlit command)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Brain Tumor Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply medical-style theme
apply_custom_css()

# -----------------------------------------------------------------------------
# Sidebar branding
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <div style="font-size: 2.5rem;">🧠</div>
        <div style="font-size: 1.1rem; font-weight: 700; color: #0077B6;">
            Brain Tumor Agent
        </div>
        <div style="font-size: 0.8rem; color: #6C757D;">
            RL + Vision Transformer
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

# -----------------------------------------------------------------------------
# Main landing page
# -----------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>Brain Tumor Agent</h1>
    <p>Reinforcement Learning Based Brain Tumor Detection using Vision Transformers</p>
</div>
""", unsafe_allow_html=True)

# How it works section
st.markdown("### How It Works")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="stat-box">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">📤</div>
        <div class="stat-label">Step 1</div>
        <div style="font-weight: 600; color: #1D3557;">Upload MRI</div>
        <div style="font-size: 0.8rem; color: #6C757D; margin-top: 0.3rem;">
            Brain MRI scan in JPG, PNG, or BMP
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stat-box">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔍</div>
        <div class="stat-label">Step 2</div>
        <div style="font-weight: 600; color: #1D3557;">ViT Analysis</div>
        <div style="font-size: 0.8rem; color: #6C757D; margin-top: 0.3rem;">
            Vision Transformer extracts features
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stat-box">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">🎯</div>
        <div class="stat-label">Step 3</div>
        <div style="font-weight: 600; color: #1D3557;">Classification</div>
        <div style="font-size: 0.8rem; color: #6C757D; margin-top: 0.3rem;">
            4-class tumor identification
        </div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="stat-box">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">💊</div>
        <div class="stat-label">Step 4</div>
        <div style="font-weight: 600; color: #1D3557;">RL Recommendation</div>
        <div style="font-size: 0.8rem; color: #6C757D; margin-top: 0.3rem;">
            DQN agent suggests treatment
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")

# Architecture diagram
st.markdown("### Architecture")

st.markdown("""
```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│  MRI Image   │────>│  ViT Encoder     │────>│  Classification  │────>│  DQN Agent         │
│  (224x224)   │     │  (768 features)  │     │  Head (4 classes)│     │  (Treatment Rec.)  │
└──────────────┘     └──────────────────┘     └──────────────────┘     └────────────────────┘
```
""")

# Tumor classes info
st.markdown("### Tumor Classes")

cls_col1, cls_col2, cls_col3, cls_col4 = st.columns(4)

class_info = [
    ("Glioma", "Tumor from glial cells", "Severity: High", "#E63946"),
    ("Meningioma", "Tumor from brain membranes", "Severity: Moderate", "#F4A261"),
    ("No Tumor", "Healthy brain scan", "Severity: None", "#2DC653"),
    ("Pituitary", "Pituitary gland tumor", "Severity: Low", "#457B9D"),
]

for col, (name, desc, sev, color) in zip(
    [cls_col1, cls_col2, cls_col3, cls_col4], class_info
):
    with col:
        st.markdown(f"""
        <div class="result-card" style="border-top: 3px solid {color};">
            <div style="font-weight: 700; color: {color}; font-size: 1rem;">{name}</div>
            <div style="color: #6C757D; font-size: 0.85rem; margin: 0.3rem 0;">{desc}</div>
            <div style="
                display: inline-block;
                background: {color}15;
                color: {color};
                padding: 0.2rem 0.6rem;
                border-radius: 10px;
                font-size: 0.75rem;
                font-weight: 600;
            ">{sev}</div>
        </div>
        """, unsafe_allow_html=True)

# Tech stack
st.markdown("### Tech Stack")
tech_cols = st.columns(5)
techs = [
    ("PyTorch", "Deep Learning Framework"),
    ("ViT", "Vision Transformer"),
    ("SB3", "Stable-Baselines3 (RL)"),
    ("DQN", "Treatment Agent"),
    ("Streamlit", "Web Interface"),
]
for col, (name, desc) in zip(tech_cols, techs):
    with col:
        st.markdown(f"""
        <div style="text-align: center; padding: 0.8rem; border: 1px solid #DEE2E6;
                    border-radius: 8px; background: white;">
            <div style="font-weight: 700; color: #0077B6;">{name}</div>
            <div style="font-size: 0.75rem; color: #6C757D;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# Disclaimer
render_disclaimer()

# Sidebar navigation hint
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Navigate** using the pages above to explore the dataset, "
    "monitor training, or run predictions."
)
