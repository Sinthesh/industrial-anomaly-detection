import streamlit as st
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Fix project imports
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

        score = result.get("score", 0)
        heatmap = np.array(result.get("heatmap", []))
        runtime = result.get("runtime_seconds", 0)

# Fix heatmap shape
if heatmap.size > 0:
    heatmap = heatmap.reshape(224, 224)

        st.write("## Inspection Result")

        col1, col2 = st.columns(2)

        with col1:
            st.write("### Anomaly Heatmap")

            fig, ax = plt.subplots()
            ax.imshow(heatmap, cmap="jet")
            ax.axis("off")

            st.pyplot(fig)

        with col2:
            st.write("### Overlay Detection")

            fig2, ax2 = plt.subplots()

            ax2.imshow(image.resize((224,224)))
            ax2.imshow(heatmap, cmap="jet", alpha=0.5)
            ax2.axis("off")

            st.pyplot(fig2)

        st.write("### Metrics")

        st.metric("Anomaly Score", round(score,3))
        st.metric("Runtime (seconds)", runtime)
