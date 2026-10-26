"""
app.py
------
Streamlit front-end for ChronoVision: an interactive demo of unsupervised
change detection on bi-temporal satellite image pairs.

The user either uploads two images or picks a sample pair from the local
`data/` folder, chooses one of the three detection methods implemented in
`src/methods.py`, and sees: the two input images, the binary change map,
a red overlay of detected change on top of Image 1, the percentage of
area changed, and -- if a ground-truth label is available for the chosen
sample -- a table of evaluation metrics.

Run with:
    streamlit run app.py

Naming note: in this LEVIR-CD mirror, folder A is technically the
post-change image and folder B the pre-change image. We deliberately
avoid "before/after" labels in the UI and instead use the neutral
"Image 1 (A)" / "Image 2 (B)" so the display never implies a temporal
order that isn't guaranteed.
"""

import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.metrics import compute_metrics
from src.methods import change_vector_analysis, image_difference_otsu, pca_kmeans
from src.postprocessing import clean_change_map
from src.preprocessing import TARGET_SIZE, histogram_match, load_image, load_label, resize_image

DATA_DIR = "data"
DIR_A = os.path.join(DATA_DIR, "A")
DIR_B = os.path.join(DATA_DIR, "B")
DIR_LABEL = os.path.join(DATA_DIR, "label")

METHODS = {
    "Image Difference + Otsu": image_difference_otsu,
    "Change Vector Analysis (CVA)": change_vector_analysis,
    "PCA + K-Means (Celik 2009)": pca_kmeans,
}

st.set_page_config(
    page_title="ChronoVision",
    page_icon="🛰️",
    layout="wide",
)


def decode_upload(uploaded_file):
    """Decode a Streamlit UploadedFile into an RGB uint8 numpy array."""
    file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Could not decode image: {uploaded_file.name}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def list_sample_names():
    """List sample filenames available in data/A that also exist in data/B."""
    if not (os.path.isdir(DIR_A) and os.path.isdir(DIR_B)):
        return []
    names_a = set(os.listdir(DIR_A))
    names_b = set(os.listdir(DIR_B))
    return sorted(names_a & names_b)


def make_overlay(base_image, change_map, color=(255, 0, 0), alpha=0.5):
    """Blend a solid `color` onto `base_image` wherever `change_map` is nonzero.

    Parameters
    ----------
    base_image : np.ndarray
        (H, W, 3) RGB uint8 image to draw the overlay on top of.
    change_map : np.ndarray
        (H, W) uint8 binary mask, nonzero = changed.
    color : tuple
        RGB color for the overlay (default: red).
    alpha : float
        Blend strength of the overlay color (0 = invisible, 1 = solid).

    Returns
    -------
    np.ndarray
        (H, W, 3) RGB uint8 image with the overlay applied.
    """
    overlay = base_image.copy().astype(np.float64)
    mask = change_map > 0
    color_arr = np.array(color, dtype=np.float64)
    overlay[mask] = (1 - alpha) * overlay[mask] + alpha * color_arr
    return overlay.astype(np.uint8)


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("ChronoVision — Change Detection in Bi-Temporal Satellite Imagery")
st.markdown(
    "Unsupervised detection of surface change between two satellite images "
    "of the same location, captured at different times."
)

# --------------------------------------------------------------------------
# Sidebar: input selection + method selection
# --------------------------------------------------------------------------
st.sidebar.header("1. Input")
input_mode = st.sidebar.radio("Choose input mode", ["Sample from data/", "Upload two images"])

img_a_raw = None
img_b_raw = None
label = None
sample_name = None

if input_mode == "Sample from data/":
    sample_names = list_sample_names()
    if not sample_names:
        st.sidebar.error(f"No sample pairs found in '{DIR_A}' / '{DIR_B}'.")
    else:
        sample_name = st.sidebar.selectbox("Choose a sample pair", sample_names)
        img_a_raw = load_image(os.path.join(DIR_A, sample_name))
        img_b_raw = load_image(os.path.join(DIR_B, sample_name))
        label_path = os.path.join(DIR_LABEL, sample_name)
        if os.path.exists(label_path):
            label = resize_image(load_label(label_path), TARGET_SIZE)
else:
    upload_a = st.sidebar.file_uploader("Image 1 (A)", type=["png", "jpg", "jpeg", "tif", "tiff"])
    upload_b = st.sidebar.file_uploader("Image 2 (B)", type=["png", "jpg", "jpeg", "tif", "tiff"])
    if upload_a is not None and upload_b is not None:
        img_a_raw = decode_upload(upload_a)
        img_b_raw = decode_upload(upload_b)

st.sidebar.header("2. Method")
method_name = st.sidebar.selectbox("Choose a change-detection method", list(METHODS.keys()))

st.sidebar.markdown("---")
st.sidebar.caption(
    "Image 1 (A) and Image 2 (B) are neutral labels for the two acquisition "
    "dates -- order does not affect the underlying change-detection math."
)

# --------------------------------------------------------------------------
# Main panel
# --------------------------------------------------------------------------
if img_a_raw is None or img_b_raw is None:
    st.info("Select a sample pair or upload two images from the sidebar to get started.")
else:
    # Preprocess: resize both to a common grid, then histogram-match B onto A.
    img_a = resize_image(img_a_raw, TARGET_SIZE)
    img_b = histogram_match(resize_image(img_b_raw, TARGET_SIZE), img_a)

    st.subheader("Input Images")
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_a, caption="Image 1 (A)", use_container_width=True)
    with col2:
        st.image(img_b, caption="Image 2 (B)", use_container_width=True)

    with st.spinner(f"Running {method_name}..."):
        change_map = clean_change_map(METHODS[method_name](img_a, img_b))

    overlay = make_overlay(img_a, change_map)
    percent_changed = 100.0 * np.count_nonzero(change_map) / change_map.size

    st.subheader("Detection Result")
    col3, col4, col5 = st.columns(3)
    with col3:
        st.image(change_map, caption="Binary Change Map", use_container_width=True, clamp=True)
    with col4:
        st.image(overlay, caption="Change Overlay on Image 1 (A)", use_container_width=True)
    with col5:
        st.metric("% of Area Changed", f"{percent_changed:.2f}%")
        st.caption(f"Method: {method_name}")
        if sample_name is not None:
            st.caption(f"Sample: {sample_name}")

    # --------------------------------------------------------------------
    # Metrics against ground truth, if available for this sample.
    # --------------------------------------------------------------------
    st.subheader("Evaluation")
    if label is not None:
        metrics = compute_metrics(change_map, label)
        metrics_df = pd.DataFrame(
            {
                "Metric": ["Precision", "Recall", "F1 Score", "IoU", "Accuracy", "Cohen's Kappa"],
                "Value": [
                    metrics["precision"],
                    metrics["recall"],
                    metrics["f1"],
                    metrics["iou"],
                    metrics["accuracy"],
                    metrics["kappa"],
                ],
            }
        )
        metrics_df["Value"] = metrics_df["Value"].map(lambda v: f"{v:.4f}")
        st.table(metrics_df.set_index("Metric"))
    else:
        st.info(
            "No ground-truth label available for this input, so quantitative "
            "metrics cannot be computed. (Metrics are shown automatically "
            "when selecting a sample from data/ that has a matching label.)"
        )
