import streamlit as st
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Fix project imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_server.controller import process_inspection


# ------------------------
# Page Config
# ------------------------

st.set_page_config(
    page_title="Industrial Defect Detection",
    layout="wide"
)

st.title("Industrial Defect Detection System")

st.write("Upload a product image and run anomaly inspection.")

# ------------------------
# Upload Section
# ------------------------

uploaded_file = st.file_uploader("Upload Image")

product = st.selectbox(
    "Select Product Type",
    ["bottle", "hazelnut", "metalnut", "pill"]
)

# ------------------------
# Run Inspection
# ------------------------

if uploaded_file is not None:

    temp_path = "temp_image.png"

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    image = Image.open(temp_path)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Run Inspection"):

        result = process_inspection(temp_path, product)

        score = result["score"]

        heatmap = np.array(result.get("heatmap"))

        st.divider()

        st.subheader("Inspection Result")

        # ------------------------
        # Status Indicator
        # ------------------------

        if score > 0.5:
            st.error(f"DEFECT DETECTED 🔴  |  Score: {score:.3f}")
        else:
            st.success(f"NORMAL PRODUCT 🟢  |  Score: {score:.3f}")

        # ------------------------
        # Visualization Section
        # ------------------------

        original = image.resize((224,224))

        col1, col2, col3 = st.columns(3)

        # Original Image
        with col1:
            st.markdown("### Original")
            st.image(original)

        # Heatmap
        with col2:
            st.markdown("### Heatmap")

            fig, ax = plt.subplots()
            ax.imshow(heatmap, cmap="jet")
            ax.axis("off")

            st.pyplot(fig)

        # Overlay
        with col3:
            st.markdown("### Overlay Detection")

            fig, ax = plt.subplots()
            ax.imshow(original)
            ax.imshow(heatmap, cmap="jet", alpha=0.5)
            ax.axis("off")

            st.pyplot(fig)

        # Runtime
        st.write(f"Runtime: **{result['runtime_seconds']} seconds**")
