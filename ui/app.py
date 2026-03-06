import streamlit as st
import sys
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mcp_server.controller import process_inspection

st.title("Industrial Defect Detection System")

uploaded_file = st.file_uploader("Upload Image")

product = st.selectbox(
    "Select Product Type",
    ["bottle", "hazelnut", "metalnut", "pill"]
)

if uploaded_file is not None:

    temp_path = "temp_image.png"

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    image = Image.open(temp_path).convert("RGB")

    st.image(image, caption="Uploaded Image", width=350)

    if st.button("Run Inspection"):

        try:
            result = process_inspection(temp_path, product)
        except Exception as e:
            st.error(f"Inspection failed: {e}")
            raise

        score = result["score"]
        heatmap = np.array(result["heatmap"], dtype=float)

        st.write("## Inspection Result")

        if score > 0.5:
            st.error(f"DEFECT DETECTED 🔴 | Score: {round(score,3)}")
        else:
            st.success(f"NORMAL PRODUCT 🟢 | Score: {round(score,3)}")

        col1, col2, col3 = st.columns(3)

        # -------------------------
        # ORIGINAL IMAGE
        # -------------------------
        with col1:
            st.subheader("Original")

            image_resized = image.resize((224, 224))
            image_np = np.array(image_resized)

            st.image(image_np, width=250)

        # -------------------------
        # HEATMAP (JET COLORMAP)
        # -------------------------
        with col2:
            st.subheader("Anomaly Heatmap")

            fig, ax = plt.subplots()

            ax.imshow(heatmap, cmap="jet")
            ax.axis("off")

            st.pyplot(fig)

        # -------------------------
        # OVERLAY (MATCH COLAB)
        # -------------------------
        with col3:
            st.subheader("Overlay Detection")

            fig2, ax2 = plt.subplots()

            ax2.imshow(image_np)
            ax2.imshow(heatmap, cmap="jet", alpha=0.45)

            ax2.axis("off")

            st.pyplot(fig2)

        # -------------------------
        # RAW JSON OUTPUT
        # -------------------------
        st.json(result)
