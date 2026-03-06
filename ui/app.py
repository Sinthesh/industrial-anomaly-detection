import streamlit as st
import sys
import os
import numpy as np
import cv2
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(**file**), "..")))
from mcp_server.controller import process_inspection

st.title("Industrial Defect Detection System")

uploaded_file = st.file_uploader("Upload Image")

product = st.selectbox(
"Select Product Type",
["bottle", "hazelnut", "metalnut", "pill"]
)

st.info(f"Model selected: {product}_padim_model.pth")

if uploaded_file is not None:

```
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

    with col1:
        st.subheader("Original")
        st.image(image, width=250)

    with col2:
        st.subheader("Anomaly Heatmap")

        # Improve visualization contrast
        heatmap_vis = heatmap.copy()

        # Clip extreme values
        heatmap_vis = np.clip(heatmap_vis, 0, np.percentile(heatmap_vis, 99))

        # Normalize again for display
        heatmap_vis = (heatmap_vis - heatmap_vis.min()) / (heatmap_vis.max() - heatmap_vis.min() + 1e-8)

        heatmap_uint8 = (heatmap_vis * 255).astype(np.uint8)

        heatmap_color = cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET
        )

        st.image(heatmap_color)

    with col3:
        st.subheader("Overlay Detection")

        image_np = np.array(image.resize((224, 224)))

        # Improve visualization contrast
        heatmap_vis = heatmap.copy()
        heatmap_vis = np.clip(heatmap_vis, 0, np.percentile(heatmap_vis, 99))
        heatmap_vis = (heatmap_vis - heatmap_vis.min()) / (heatmap_vis.max() - heatmap_vis.min() + 1e-8)

        heatmap_uint8 = (heatmap_vis * 255).astype(np.uint8)

        heatmap_color = cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET
        )

        overlay = cv2.addWeighted(
            image_np,
            0.7,
            heatmap_color,
            0.3,
            0
        )

        st.image(overlay)

    st.json(result)
```
