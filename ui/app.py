import streamlit as st
import sys
import os
import numpy as np
import cv2
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mcp_server.controller import process_inspection

st.title("Industrial Defect Detection System")

uploaded_file = st.file_uploader("Upload Image")
product = st.selectbox(
    "Select Product Type", ["bottle", "hazelnut", "metalnut", "pill"]
)

if uploaded_file is not None:
    temp_path = "temp_image.png"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    image = Image.open(temp_path)
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

        with col1:
            st.subheader("Original")
            st.image(image, width=250)

        with col2:
            st.subheader("Anomaly Heatmap")

            heatmap_uint8 = (heatmap * 255).astype(np.uint8)

            heatmap_color = cv2.applyColorMap(
                heatmap_uint8,
                cv2.COLORMAP_JET
            )

            st.image(heatmap_color)

        with col3:
            st.subheader("Overlay")
            image_np = np.array(image.resize((224,224)))
            heatmap_uint8 = (heatmap * 255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(
                heatmap_uint8, cv2.COLORMAP_JET
            )
            overlay = cv2.addWeighted(
                image_np, 0.6, heatmap_color, 0.4, 0
            )
            st.image(overlay)

        st.json(result)
