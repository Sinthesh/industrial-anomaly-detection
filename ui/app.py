import streamlit as st
import sys
import os
import numpy as np
import cv2
from PIL import Image

# Allow project imports
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

        # ------------------------
        # Original Image
        # ------------------------

        with col1:
            st.subheader("Original")
            st.image(image, width=250)

        # ------------------------
        # Heatmap
        # ------------------------

        with col2:
            st.subheader("Heatmap")
            st.image((heatmap * 255).astype("uint8"))

        # ------------------------
        # Overlay + Bounding Box
        # ------------------------

        with col3:

            st.subheader("Overlay + Defect Box")

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

            # ------------------------
            # Defect Localization
            # ------------------------

            threshold = 0.6
            mask = heatmap > threshold

            coords = np.column_stack(np.where(mask))

            if len(coords) > 0:

                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)

                cv2.rectangle(
                    overlay,
                    (x_min, y_min),
                    (x_max, y_max),
                    (255, 0, 0),
                    2
                )

            st.image(overlay)

        # ------------------------
        # JSON Result
        # ------------------------

        st.json(result)
