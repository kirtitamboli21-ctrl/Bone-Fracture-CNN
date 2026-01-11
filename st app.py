
# Streamlit version of the X-ray Fracture Detection App
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# X-ray inspired color palette (light, clinical, blue/cyan/white)
CLR_BG = "#2E5897"         # X-ray film white
CLR_CARD = "#000000"       # Soft blue card
CLR_ACCENT = "#00B4D8"     # X-ray cyan
CLR_ACCENT2 = "#48CAE4"    # Lighter cyan
CLR_TEXT = "#FDFDFD"       # Deep gray text
CLR_SUBTLE = "#B0BEC5"     # Subtle gray
CLR_DANGER = "#FF6B6B"     # Alert red
CLR_SUCCESS = "#43D19E"    # Green for negative
CLR_WARNING = "#FFD166"    # Amber for highlights

IMG_SIZE = (180, 180)
MODEL_PATH = 'fracture_classification_model.keras'

TRAIN_TOTAL = 8863
TRAIN_FRACTURED = 4480
TRAIN_NOT_FRACTURED = 4383
TEST_TOTAL = 600
TEST_FRACTURED = 360
TEST_NOT_FRACTURED = 240
CLASSES = ['fractured', 'not_fractured']
VAL_TOTAL = int(TRAIN_TOTAL * 0.2)

# Set Streamlit page config and force light theme
st.set_page_config(
    page_title="Bone Fracture Detection using CNN",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🦴"
)


# Custom CSS for X-ray theme (improved)
st.markdown(f"""
    <style>
    .stApp {{
        background-color: {CLR_BG};
    }}
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}
    .card {{
        background: {CLR_CARD};
        border-radius: 16px;
        padding: 1.5rem 1.5rem 1.2rem 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 2px 8px #e3f0ff44;
    }}
    .stat-title {{
        color: {CLR_ACCENT2};
        font-weight: bold;
        font-size: 1.1em;
    }}
    .stat-value {{
        color: {CLR_TEXT};
        font-size: 2.1em;
        font-weight: bold;
    }}
    .stat-desc {{
        color: {CLR_SUBTLE};
        font-size: 0.95em;
    }}
    </style>
""", unsafe_allow_html=True)


# Sidebar (improved)
with st.sidebar:
    st.markdown(f"<h2 style='color:{CLR_ACCENT};margin-bottom:0.2em;'>🦴 FRACTURE FLOW</h2>", unsafe_allow_html=True)
    st.markdown(f"<span style='color:{CLR_SUBTLE};font-size:1.1em;'>Diagnostic Suite</span>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigation", ["Dashboard", "X-ray Scanner", "System Specs"],
                   label_visibility="collapsed")
    st.markdown(f"<div style='margin-top:2em;color:{CLR_SUBTLE};font-size:0.95em;'>&copy; 2026 XrayAI</div>", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.warning("Model not found. Running in UI-Preview mode.")
        return None
model = load_model()


def dashboard():
    st.markdown(f"<h1 style='color:{CLR_ACCENT};margin-bottom:0.5em;'>Clinical Dashboard</h1>", unsafe_allow_html=True)
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
                <div class='card' style='text-align:center;'>
                    <div class='stat-title' style='color:{CLR_SUCCESS};'>Test Accuracy</div>
                    <div class='stat-value'>95.0%</div>
                    <div class='stat-desc'>Performance of the model on unseen X-ray images.</div>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div class='card' style='text-align:center;'>
                    <div class='stat-title' style='color:{CLR_ACCENT2};'>Training Accuracy</div>
                    <div class='stat-value'>91.5%</div>
                    <div class='stat-desc'>Accuracy achieved during model training.</div>
                </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
                <div class='card' style='text-align:center;'>
                    <div class='stat-title' style='color:{CLR_DANGER};'>Validation Loss</div>
                    <div class='stat-value'>1.96</div>
                    <div class='stat-desc'>Indicates overfitting if much higher than training loss.</div>
                </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
                <div class='card' style='text-align:center;'>
                    <div class='stat-title' style='color:{CLR_WARNING};'>Latency</div>
                    <div class='stat-value'>120ms</div>
                    <div class='stat-desc'>Average time to analyze an X-ray.</div>
                </div>
            """, unsafe_allow_html=True)

    with st.container():
        st.markdown(f"""
            <div class='card'>
                <h3 style='color:{CLR_ACCENT};margin-bottom:0.5em;'>Model Overview</h3>
                <span style='color:{CLR_TEXT};font-size:1.1em;'>
                • CNN with 3 Conv layers, 1 Dense, Dropout<br>
                • Input: 180x180 RGB X-ray<br>
                • Augmentation: Shear, Zoom, Flip<br>
                • Optimizer: Adam (1e-4) | Loss: Binary Cross-entropy
                </span>
            </div>
        """, unsafe_allow_html=True)

    with st.container():
        st.markdown(f"""
            <div class='card'>
                <h3 style='color:{CLR_ACCENT2};margin-bottom:0.5em;'>Dataset Summary</h3>
                <span style='color:{CLR_TEXT};font-size:1.1em;'>
                • Training images: {TRAIN_TOTAL}<br>
                &nbsp;&nbsp;&nbsp;&nbsp;- fractured: {TRAIN_FRACTURED}<br>
                &nbsp;&nbsp;&nbsp;&nbsp;- not_fractured: {TRAIN_NOT_FRACTURED}<br>
                • Validation (est.): {VAL_TOTAL}<br>
                • Test images: {TEST_TOTAL}<br>
                &nbsp;&nbsp;&nbsp;&nbsp;- fractured: {TEST_FRACTURED}<br>
                &nbsp;&nbsp;&nbsp;&nbsp;- not_fractured: {TEST_NOT_FRACTURED}<br>
                • Classes: {', '.join(CLASSES)}
                </span>
            </div>
        """, unsafe_allow_html=True)

    with st.container():
        st.markdown(f"<h4 style='color:{CLR_ACCENT};margin-top:1.5em;'>Instructions:</h4>", unsafe_allow_html=True)
        st.markdown(f"<span style='color:{CLR_TEXT};font-size:1.1em;'>"
                    "1. Use the X-ray Scanner to upload and analyze images.<br>"
                    "2. Review model details and system specs for transparency.<br>"
                    "3. For best results, use clear, high-resolution X-rays."
                    "</span>", unsafe_allow_html=True)


def scanner():
    st.markdown(f"<h1 style='color:{CLR_ACCENT};margin-bottom:0.5em;'>X-ray Scanner</h1>", unsafe_allow_html=True)
    with st.container():
        uploaded_file = st.file_uploader("Upload an X-ray image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            img_raw = Image.open(uploaded_file)
            st.image(img_raw, caption="Uploaded X-ray", width=450)
            if model:
                img = img_raw.convert('RGB').resize(IMG_SIZE)
                img_arr = np.array(img) / 255.0
                img_arr = np.expand_dims(img_arr, axis=0)
                pred = model.predict(img_arr)[0][0]
                pred_percent = pred * 100
                not_fractured_percent = pred_percent
                fractured_percent = 100 - pred_percent
                details = (
                    f"Raw model output: {pred:.6f}\n"
                    f"Probability Not Fractured: {not_fractured_percent:.2f}%\n"
                    f"Probability Fractured: {fractured_percent:.2f}%\n"
                    f"Threshold: 0.5\n"
                    f"File: {uploaded_file.name}\n"
                    f"Image size: {img_raw.size[0]}x{img_raw.size[1]} (original), {IMG_SIZE[0]}x{IMG_SIZE[1]} (model input)"
                )
                st.markdown(f"<div class='card'><pre style='color:black;background:{CLR_CARD};border-radius:10px;padding:10px;margin-bottom:0;'>{details}</pre></div>", unsafe_allow_html=True)
                if pred > 0.5:
                    st.success(f"NEGATIVE ({pred*100:.1f}%)", icon="✅")
                    st.progress(float(pred), text=f"Confidence: <span style='color:{CLR_TEXT};'>{{:.1f}}% Not Fractured</span>".format(float(pred)*100))
                else:
                    st.error(f"FRACTURE DETECTED ({(1-pred)*100:.1f}%)", icon="⚠️")
                    st.progress(float(1-pred), text=f"Confidence: <span style='color:{CLR_TEXT};'>{{:.1f}}% Fractured</span>".format(float(1-pred)*100))
            else:
                st.warning("Model not loaded. Cannot run inference.")
        else:
            st.info("Please upload an X-ray image to analyze.")


def model_details():
    st.markdown(f"<h1 style='color:{CLR_ACCENT};margin-bottom:0.5em;'>System Specs</h1>", unsafe_allow_html=True)
    specs = [
        ("Optimizer", "Adam (Learning Rate: 1x10⁻⁴)"),
        ("Loss Function", "Binary Cross-entropy"),
        ("Layers", "Conv2D (32, 64, 128) -> Dense (128)"),
        ("Augmentation", "Shear/Zoom 20%, Horizontal Flip"),
        ("Validation Loss", "1.96 (Indicates Overfitting)")
    ]
    for label, val in specs:
        st.markdown(f"""
            <div class='card' style='display:flex;justify-content:space-between;align-items:center;'>
                <span style='color:{CLR_ACCENT2};font-weight:bold;font-size:1.1em;'>{label}</span>
                <span style='color:{CLR_TEXT};font-size:1.1em;'>{val}</span>
            </div>
        """, unsafe_allow_html=True)

if page == "Dashboard":
    dashboard()
elif page == "X-ray Scanner":
    scanner()
else:
    model_details()
