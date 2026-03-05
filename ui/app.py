import streamlit as st
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_server.controller import process_inspection

st.title("Industrial Defect Detection System")

st.write("Upload a product image and run anomaly inspection.")

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

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Run Inspection"):

        result = process_inspection(temp_path, product)

        score = result["score"]
        heatmap = np.array(result["heatmap"])

        st.write("## Inspection Result")
        st.write(f"Anomaly Score: **{score:.3f}**")

        # Resize original for overlay
        original = image.resize((224,224))

        col1, col2, col3 = st.columns(3)

        # Original
        with col1:
            st.subheader("Original")
            st.image(original)

        # Heatmap
        with col2:
            fig, ax = plt.subplots()
            ax.imshow(heatmap, cmap="jet")
            ax.axis("off")
            st.pyplot(fig)

        # Overlay
        with col3:
            fig, ax = plt.subplots()
            ax.imshow(original)
            ax.imshow(heatmap, cmap="jet", alpha=0.5)
            ax.axis("off")
            st.pyplot(fig)

        st.write("Runtime:", result["runtime_seconds"], "seconds")
