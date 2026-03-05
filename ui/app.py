import streamlit as st
import sys
import os
import numpy as np
import cv2
from PIL import Image

# Allow project imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mcp_server.controller import process_inspection


# -----------------------------
# Page Title
# -----------------------------

st.title("Industrial Defect Detection System")
st.write("Upload a product image and run anomaly inspection.")


# -----------------------------
# Upload Image
# -----------------------------

uploaded_file = st.file_uploader("Upload Image")

product = st.selectbox(
    "Select Product Type",
    ["bottle", "hazelnut", "metalnut", "pill"]
)


# -----------------------------
# When Image Uploaded
# -----------------------------

if uploaded_file is not None:

    temp_path = "temp_image.png"

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    image = Image.open(temp_path)

    st.image(image, caption="Uploaded Image", width=350)


    # -----------------------------
    # Run Inspection Button
    # -----------------------------

    if st.button("Run Inspection"):

        result = process_inspection(temp_path, product)

        score = result["score"]
        heatmap = np.array(result["heatmap"], dtype=float)

        st.write("## Inspection Result")

        # -----------------------------
        # Defect Status
        # -----------------------------

        if score > 0.5:
            st.error(f"DEFECT DETECTED 🔴 | Score: {round(score,3)}")
        else:
            st.success(f"NORMAL PRODUCT 🟢 | Score: {round(score,3)}")


        # -----------------------------
        # Visualization Layout
        # -----------------------------

        col1, col2, col3 = st.columns(3)


        # -----------------------------
        # Original Image
        # -----------------------------

        with col1:
            st.subheader("Original")
            st.image(image, width=250)


        # -----------------------------
        # Heatmap
        # -----------------------------

        with col2:
            st.subheader("Heatmap")

            st.image(
                heatmap,
                caption="Anomaly Heatmap",
                clamp=True
            )


        # -----------------------------
        # Overlay Detection
        # -----------------------------

        with col3:

            st.subheader("Overlay")

            image_np = np.array(image.resize((224,224)))

            heatmap_uint8 = (heatmap * 255).astype(np.uint8)

            heatmap_color = cv2.applyColorMap(
                heatmap_uint8,
                cv2.COLORMAP_JET
            )

            overlay = cv2.addWeighted(
                image_np,
                0.6,
                heatmap_color,
                0.4,
                0
            )

            st.image(overlay, caption="Overlay Detection")


        # -----------------------------
        # Raw Result JSON
        # -----------------------------

        st.write("### Inspection Data")
        st.json(result)
