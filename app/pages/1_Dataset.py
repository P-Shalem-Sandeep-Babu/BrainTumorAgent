"""
Page 1: Dataset Viewer.

Browse and explore brain tumor MRI images from the dataset.
Shows sample images from each class with basic statistics.
"""

import streamlit as st
from pathlib import Path
from app.utils.ui_components import apply_custom_css, render_header, render_disclaimer

st.set_page_config(page_title="Dataset Viewer", layout="wide")
apply_custom_css()

render_header("Dataset Viewer", "Browse brain tumor MRI images organized by class.")

# --- Dataset Statistics ---
st.header("Dataset Overview")

DATA_DIR = Path("data/raw")
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

cols = st.columns(len(CLASS_NAMES))
for i, class_name in enumerate(CLASS_NAMES):
    class_dir = DATA_DIR / class_name
    if class_dir.exists():
        count = len(list(class_dir.glob("*")))
    else:
        count = 0
    with cols[i]:
        st.metric(label=class_name.capitalize(), value=f"{count} images")

# --- Image Browser ---
st.header("Browse Images")

selected_class = st.selectbox("Select a class", CLASS_NAMES)
class_dir = DATA_DIR / selected_class

if class_dir.exists():
    images = sorted(list(class_dir.glob("*")))
    if images:
        # Show a slider to pick how many to display
        n_show = st.slider("Number of images to display", 1, min(20, len(images)), 4)
        cols = st.columns(min(n_show, 4))
        for i in range(n_show):
            with cols[i % 4]:
                st.image(str(images[i]), caption=images[i].name, use_container_width=True)
    else:
        st.warning(f"No images found in {class_dir}")
else:
    st.warning(f"Directory not found: {class_dir}")
    st.info("Download a brain tumor MRI dataset and place it in `data/raw/`.")

render_disclaimer()
