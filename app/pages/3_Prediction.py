"""
Page 3: Prediction.

Upload a brain MRI scan and get:
1. Tumor type classification (from ViT)
2. Confidence score with visualization
3. Treatment recommendation (from DQN agent)
4. Prediction history tracking

Features:
- Medical-style UI with clean design
- Loading animations during inference
- Confidence bar chart and donut chart
- Severity indicators
- Prediction history sidebar
- Error handling for missing models
"""

import streamlit as st
from PIL import Image

from app.utils.predictor import get_predictor
from app.utils.ui_components import (
    apply_custom_css,
    render_header,
    render_confidence_chart,
    render_donut_chart,
    render_tumor_result,
    render_recommendation_card,
    render_disclaimer,
    render_loading_spinner,
    init_history,
    add_to_history,
    render_history,
    clear_history,
    COLORS,
)

# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Prediction - Brain Tumor Agent", layout="wide")
apply_custom_css()

# Initialize history
init_history()

# -----------------------------------------------------------------------------
# Sidebar: Model status and history
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Model Status")
    predictor = get_predictor()
    model_info = predictor.get_model_info()

    # ViT status
    if model_info["vit_available"]:
        st.success("ViT Classifier: Ready")
        if "vit_val_acc" in model_info:
            st.caption(f"Val Accuracy: {model_info['vit_val_acc']:.2%}")
    else:
        st.error("ViT Classifier: Not Found")
        st.caption(f"Expected at: `{model_info['vit_path']}`")

    # DQN status
    if model_info["dqn_available"]:
        st.success("DQN Agent: Ready")
    else:
        st.warning("DQN Agent: Not Found")
        st.caption("Using rule-based fallback")

    st.caption(f"Device: {model_info['device']}")

    st.markdown("---")

    # Prediction history
    st.markdown("### Prediction History")
    render_history()

    if st.session_state.prediction_history:
        if st.button("Clear History", type="secondary"):
            clear_history()
            st.rerun()

# -----------------------------------------------------------------------------
# Main content
# -----------------------------------------------------------------------------
render_header(
    "Tumor Prediction",
    "Upload a brain MRI scan to classify the tumor type and get a treatment recommendation.",
)

# -----------------------------------------------------------------------------
# File uploader
# -----------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "jpeg", "png", "bmp"],
    help="Upload a brain MRI scan in JPG, PNG, or BMP format.",
    label_visibility="collapsed",
)

# -----------------------------------------------------------------------------
# Prediction flow
# -----------------------------------------------------------------------------
if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file)

    # Layout: Image on left, results on right
    col_img, col_results = st.columns([1, 1])

    with col_img:
        st.markdown("#### Uploaded MRI")
        st.image(image, caption=uploaded_file.name, use_container_width=True)

        # Image info
        st.markdown(f"""
        <div style="
            background: #F8F9FA;
            border-radius: 8px;
            padding: 0.8rem 1rem;
            font-size: 0.85rem;
            color: #6C757D;
        ">
            <strong>File:</strong> {uploaded_file.name}<br>
            <strong>Size:</strong> {image.size[0]} x {image.size[1]} px<br>
            <strong>Mode:</strong> {image.mode}
        </div>
        """, unsafe_allow_html=True)

    with col_results:
        # Run prediction with loading animation
        with render_loading_spinner("Analyzing MRI scan..."):
            try:
                result = predictor.predict(image)
            except RuntimeError as e:
                st.error(f"Prediction failed: {e}")
                st.info(
                    "Please train the ViT model first. "
                    "Go to the **Training** page or run `python train.py`."
                )
                st.stop()
            except Exception as e:
                st.error(f"Unexpected error during prediction: {e}")
                st.stop()

        # Add to history
        add_to_history(result, uploaded_file.name)

        # Source summary banner
        vit_src = result.get("vit_source", "Trained ViT")
        rec_src = result.get("rec_source", "DQN Agent")
        if result.get("demo_mode"):
            st.warning(
                "**Demo Mode** — No trained models found. "
                "Results are simulated based on image properties. "
                "Train locally with `python train.py` for real predictions."
            )
        elif rec_src == "Rule-based":
            st.info(
                f"**Classification:** {vit_src} | "
                f"**Recommendation:** {rec_src} (DQN model not found)"
            )

        # ---- Tumor Classification Result ----
        st.markdown("#### Classification Result")
        render_tumor_result(
            tumor_type=result["tumor_type"],
            confidence=result["confidence"],
            severity=result["severity"],
            source=result.get("vit_source", "Trained ViT"),
        )

        # ---- Treatment Recommendation ----
        st.markdown("#### Treatment Recommendation")
        render_recommendation_card(
            recommendation=result["recommendation"],
            action_index=result["recommendation_index"],
            source=result.get("rec_source", "DQN Agent"),
        )

    # ---- Detailed Analysis (full width below) ----
    st.markdown("---")
    st.markdown("### Detailed Analysis")

    chart_col1, chart_col2 = st.columns([2, 1])

    with chart_col1:
        st.markdown("#### Confidence by Class")
        fig_bar = render_confidence_chart(
            result["probabilities"],
            result["tumor_type"],
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with chart_col2:
        st.markdown("#### Probability Distribution")
        fig_donut = render_donut_chart(result["probabilities"])
        st.plotly_chart(fig_donut, use_container_width=True)

    # ---- Class Details Table ----
    st.markdown("#### All Class Probabilities")

    table_data = []
    for cls in result["all_classes"]:
        severity_map = {"glioma": 3, "meningioma": 2, "pituitary": 1, "notumor": 0}
        sev = severity_map.get(cls["name"], 0)
        table_data.append({
            "Class": cls["name"].capitalize(),
            "Confidence": f"{cls['probability']:.2%}",
            "Severity": sev,
            "Status": "Predicted" if cls["name"] == result["tumor_type"] else "",
        })

    st.dataframe(
        table_data,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Class": st.column_config.TextColumn("Tumor Class"),
            "Confidence": st.column_config.TextColumn("Confidence"),
            "Severity": st.column_config.NumberColumn("Severity Level"),
            "Status": st.column_config.TextColumn("Result"),
        },
    )

    # Disclaimer
    render_disclaimer()

else:
    # ---- Empty state ----
    st.markdown("""
    <div style="
        text-align: center;
        padding: 4rem 2rem;
        background: #F8F9FA;
        border-radius: 12px;
        border: 2px dashed #DEE2E6;
        margin-top: 1rem;
    ">
        <div style="font-size: 3rem; margin-bottom: 1rem;">🧠</div>
        <h3 style="color: #1D3557; margin-bottom: 0.5rem;">Upload an MRI Scan</h3>
        <p style="color: #6C757D; max-width: 400px; margin: 0 auto;">
            Drag and drop a brain MRI image above, or click to browse.
            Supported formats: JPG, JPEG, PNG, BMP.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Example workflow
    st.markdown("### What Happens After Upload?")

    step_cols = st.columns(3)
    steps = [
        ("1. Image Preprocessing", "The MRI scan is resized to 224x224 pixels and normalized using ImageNet statistics for the ViT model."),
        ("2. ViT Classification", "The Vision Transformer extracts 768-dimensional features and classifies the tumor into one of 4 types with a confidence score."),
        ("3. RL Recommendation", "The DQN agent receives the tumor type and confidence, then recommends an appropriate treatment action."),
    ]

    for col, (title, desc) in zip(step_cols, steps):
        with col:
            st.markdown(f"""
            <div class="result-card">
                <div style="font-weight: 600; color: #0077B6; margin-bottom: 0.5rem;">{title}</div>
                <div style="color: #6C757D; font-size: 0.9rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
