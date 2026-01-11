
# Streamlit version of the X-ray Fracture Detection App
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# X-ray inspired color palette (light, clinical, blue/cyan/white)
CLR_BG = "#F6F8FB"         # X-ray film white
CLR_CARD = "#E3F0FF"       # Soft blue card
CLR_ACCENT = "#00B4D8"     # X-ray cyan
CLR_ACCENT2 = "#48CAE4"    # Lighter cyan
CLR_TEXT = "#222831"       # Deep gray text
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

# Set Streamlit page config
st.set_page_config(
    page_title="FractureFlow AI v2.0",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🦴"
)

# Custom CSS for X-ray theme
st.markdown(f"""
    <style>
    .stApp {{
        background-color: {CLR_BG};
    }}
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown(f"<h2 style='color:{CLR_ACCENT};'>FRACTURE FLOW</h2>", unsafe_allow_html=True)
st.sidebar.markdown(f"<span style='color:{CLR_SUBTLE};'>Diagnostic Suite</span>", unsafe_allow_html=True)
page = st.sidebar.radio("Navigation", ["Dashboard", "X-ray Scanner", "System Specs"])
st.sidebar.markdown(f"<hr><span style='color:{CLR_SUBTLE};'>&copy; 2026 XrayAI</span>", unsafe_allow_html=True)

# Load Model
@st.cache_resource(allow_output_mutation=True)
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.warning("Model not found. Running in UI-Preview mode.")
        return None
model = load_model()

def dashboard():
    st.markdown(f"<h1 style='color:{CLR_ACCENT};'>Clinical Dashboard</h1>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div style='background-color:{CLR_CARD};border-radius:15px;padding:20px;text-align:center;'>"
                    f"<span style='color:{CLR_SUCCESS};font-weight:bold;'>Test Accuracy</span><br>"
                    f"<span style='font-size:2em;color:{CLR_TEXT};'>95.0%</span><br>"
                    f"<span style='color:{CLR_SUBTLE};font-size:0.9em;'>Performance of the model on unseen X-ray images.</span>"
                    f"</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div style='background-color:{CLR_CARD};border-radius:15px;padding:20px;text-align:center;'>"
                    f"<span style='color:{CLR_ACCENT2};font-weight:bold;'>Training Accuracy</span><br>"
                    f"<span style='font-size:2em;color:{CLR_TEXT};'>91.5%</span><br>"
                    f"<span style='color:{CLR_SUBTLE};font-size:0.9em;'>Accuracy achieved during model training.</span>"
                    f"</div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div style='background-color:{CLR_CARD};border-radius:15px;padding:20px;text-align:center;'>"
                    f"<span style='color:{CLR_DANGER};font-weight:bold;'>Validation Loss</span><br>"
                    f"<span style='font-size:2em;color:{CLR_TEXT};'>1.96</span><br>"
                    f"<span style='color:{CLR_SUBTLE};font-size:0.9em;'>Indicates overfitting if much higher than training loss.</span>"
                    f"</div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div style='background-color:{CLR_CARD};border-radius:15px;padding:20px;text-align:center;'>"
                    f"<span style='color:{CLR_WARNING};font-weight:bold;'>Latency</span><br>"
                    f"<span style='font-size:2em;color:{CLR_TEXT};'>120ms</span><br>"
                    f"<span style='color:{CLR_SUBTLE};font-size:0.9em;'>Average time to analyze an X-ray.</span>"
                    f"</div>", unsafe_allow_html=True)

    st.markdown(f"<div style='background-color:{CLR_CARD};border-radius:15px;padding:20px;margin-top:20px;'>"
                f"<h3 style='color:{CLR_ACCENT};'>Model Overview</h3>"
                f"<span style='color:{CLR_TEXT};font-size:1.1em;'>"
                f"• CNN with 3 Conv layers, 1 Dense, Dropout<br>"
                f"• Input: 180x180 RGB X-ray<br>"
                f"• Augmentation: Shear, Zoom, Flip<br>"
                f"• Optimizer: Adam (1e-4) | Loss: Binary Cross-entropy"
                f"</span></div>", unsafe_allow_html=True)

    st.markdown(f"<div style='background-color:{CLR_CARD};border-radius:15px;padding:20px;margin-top:20px;'>"
                f"<h3 style='color:{CLR_ACCENT2};'>Dataset Summary</h3>"
                f"<span style='color:{CLR_TEXT};font-size:1.1em;'>"
                f"• Training images: {TRAIN_TOTAL}<br>"
                f"&nbsp;&nbsp;&nbsp;&nbsp;- fractured: {TRAIN_FRACTURED}<br>"
                f"&nbsp;&nbsp;&nbsp;&nbsp;- not_fractured: {TRAIN_NOT_FRACTURED}<br>"
                f"• Validation (est.): {VAL_TOTAL}<br>"
                f"• Test images: {TEST_TOTAL}<br>"
                f"&nbsp;&nbsp;&nbsp;&nbsp;- fractured: {TEST_FRACTURED}<br>"
                f"&nbsp;&nbsp;&nbsp;&nbsp;- not_fractured: {TEST_NOT_FRACTURED}<br>"
                f"• Classes: {', '.join(CLASSES)}"
                f"</span></div>", unsafe_allow_html=True)

    st.markdown(f"<h4 style='color:{CLR_ACCENT};margin-top:30px;'>Instructions:</h4>"
                f"<span style='color:{CLR_TEXT};font-size:1.1em;'>"
                f"1. Use the X-ray Scanner to upload and analyze images.<br>"
                f"2. Review model details and system specs for transparency.<br>"
                f"3. For best results, use clear, high-resolution X-rays."
                f"</span>", unsafe_allow_html=True)

def scanner():
    st.markdown(f"<h1 style='color:{CLR_ACCENT};'>X-ray Scanner</h1>", unsafe_allow_html=True)
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
            st.markdown(f"<pre style='color:{CLR_SUBTLE};background:{CLR_CARD};border-radius:10px;padding:10px;'>{details}</pre>", unsafe_allow_html=True)
            if pred > 0.5:
                st.success(f"NEGATIVE ({pred*100:.1f}%)")
                st.progress(pred)
            else:
                st.error(f"FRACTURE DETECTED ({(1-pred)*100:.1f}%)")
                st.progress(1-pred)
        else:
            st.warning("Model not loaded. Cannot run inference.")
    else:
        st.info("Please upload an X-ray image to analyze.")

def model_details():
    st.markdown(f"<h1 style='color:{CLR_ACCENT};'>System Specs</h1>", unsafe_allow_html=True)
    specs = [
        ("Optimizer", "Adam (Learning Rate: 1x10⁻⁴)"),
        ("Loss Function", "Binary Cross-entropy"),
        ("Layers", "Conv2D (32, 64, 128) -> Dense (128)"),
        ("Augmentation", "Shear/Zoom 20%, Horizontal Flip"),
        ("Validation Loss", "1.96 (Indicates Overfitting)")
    ]
    for label, val in specs:
        st.markdown(f"<div style='background-color:{CLR_CARD};border-radius:10px;padding:15px;margin-bottom:10px;display:flex;justify-content:space-between;'>"
                    f"<span style='color:{CLR_ACCENT2};font-weight:bold;'>{label}</span>"
                    f"<span style='color:{CLR_TEXT};'>{val}</span>"
                    f"</div>", unsafe_allow_html=True)

if page == "Dashboard":
    dashboard()
elif page == "X-ray Scanner":
    scanner()
else:
    model_details()
