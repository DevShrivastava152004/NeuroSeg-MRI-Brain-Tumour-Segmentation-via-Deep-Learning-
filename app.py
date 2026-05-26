import streamlit as st
import torch
import numpy as np
import segmentation_models_pytorch as smp
from PIL import Image

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="NEUROSEG - AI MRI Tumor Analysis Platform",
    page_icon="🧠",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #0b1120;
}

h1, h2, h3 {
    color: white;
}

[data-testid="stMetricValue"] {
    font-size: 35px;
    color: #00ff99;
}

[data-testid="stMetricLabel"] {
    font-size: 18px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE SECTION
# =========================================================

st.title("🧠 NEUROSEG - AI MRI Tumor Analysis Platform")

st.markdown("""
Upload a **Brain MRI Scan** and the AI model will analyze
tumor regions using a **Deep Learning U-Net segmentation model**.
""")

st.markdown("---")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📌 About")

st.sidebar.info(
    """
This project uses:

✅ U-Net Architecture  
✅ Deep Learning  
✅ Computer Vision  
✅ MRI Image Segmentation  

to identify tumor regions from MRI scans.
"""
)

st.sidebar.markdown("---")

st.sidebar.success(
    "Model: U-Net + ResNet34"
)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = smp.Unet(

        encoder_name="resnet34",

        encoder_weights=None,

        in_channels=3,

        classes=1,

        activation=None
    )

    model.load_state_dict(
        torch.load(
            "best_model.pth",
            map_location="cpu"
        )
    )

    model.eval()

    return model

model = load_model()

# =========================================================
# IMAGE SETTINGS
# =========================================================

IMG_SIZE = 256

# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def preprocess_image(image):

    image = image.resize((IMG_SIZE, IMG_SIZE))

    image = np.array(image)

    image = image / 255.0

    return image

# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict(image):

    # ---------------------------------
    # PREPROCESS IMAGE
    # ---------------------------------

    image = preprocess_image(image)

    tensor = torch.tensor(
        image,
        dtype=torch.float32
    )

    tensor = tensor.permute(2, 0, 1)

    tensor = tensor.unsqueeze(0)

    # ---------------------------------
    # MODEL PREDICTION
    # ---------------------------------

    with torch.no_grad():

        output = model(tensor)

        output = torch.sigmoid(output)

    prediction = output.squeeze().numpy()

    # ---------------------------------
    # BINARY MASK
    # ---------------------------------

    binary_mask = (
        prediction > 0.5
    ).astype(np.uint8)

    # ---------------------------------
    # VISIBLE MASK
    # ---------------------------------

    visible_mask = binary_mask * 255

    # ---------------------------------
    # OVERLAY CREATION
    # ---------------------------------

    overlay = image.copy()

    red_mask = np.zeros_like(image)

    red_mask[:, :, 0] = binary_mask

    alpha = 0.5

    overlay = (
        (1 - alpha) * image +
        alpha * red_mask
    )

    overlay = np.clip(
        overlay,
        0,
        1
    )

    # ---------------------------------
    # DETECTED REGION %
    # ---------------------------------

    tumor_pixels = np.sum(binary_mask)

    total_pixels = binary_mask.size

    tumor_percentage = (
        tumor_pixels / total_pixels
    ) * 100

    # ---------------------------------
    # CONFIDENCE SCORE
    # ---------------------------------

    if tumor_pixels > 0:

        confidence = np.mean(
            prediction[binary_mask == 1]
        ) * 100

    else:

        confidence = 0

    return (
        image,
        visible_mask,
        overlay,
        tumor_percentage,
        confidence
    )

# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📤 Upload MRI Image",
    type=["png", "jpg", "jpeg", "tif"]
)

# =========================================================
# RUN PREDICTION
# =========================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    with st.spinner("🔍 Analyzing MRI scan..."):

        (
            original,
            mask,
            overlay,
            tumor_percentage,
            confidence
        ) = predict(image)

    st.success("✅ Analysis Completed Successfully")

    st.markdown("---")

    # =====================================================
    # METRICS SECTION
    # =====================================================

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Detected Tumor Region",
            f"{tumor_percentage:.2f}%"
        )

    with col2:

        st.metric(
            "Confidence Score",
            f"{confidence:.2f}%"
        )

    st.markdown("---")

    # =====================================================
    # IMAGE DISPLAY
    # =====================================================

    col1, col2, col3 = st.columns(3)

    with col1:

        st.image(
            original,
            caption="🧠 Original MRI",
            use_column_width=True
        )

    with col2:

        st.image(
            mask,
            caption="📍 Predicted Tumor Mask",
            use_column_width=True
        )

    with col3:

        st.image(
            overlay,
            caption="🔴 Tumor Overlay",
            use_column_width=True
        )

    st.markdown("---")

    # =====================================================
    # HOW IT WORKS
    # =====================================================

    st.subheader("⚙️ How It Works")

    st.markdown("""
1. Upload a Brain MRI image  
2. AI model preprocesses the image  
3. U-Net segmentation predicts tumor region  
4. Tumor mask and overlay are generated  
5. Confidence score and detected region are displayed  
""")