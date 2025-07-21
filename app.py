
import streamlit as st
from model_function import prediction
import time
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Image Classification", layout="centered")


st.title("Image Classification App")
st.write("Upload an image of animal (🐱Cat and 🐶Dog) and let the model predict what it is!")
st.write("Model accuracy: 67%")
# Initialize photo state
if "photo" not in st.session_state:
    st.session_state["photo"] = "not done"

def change_photo_state():
    st.session_state["photo"] = "done"


with st.sidebar:
    st.header("Instructions")
    st.markdown("""
    - Upload an image file (e.g., JPG, PNG).
    - Click the 'Classify Image' button.
    - Wait for the model to process and return the result.
    """)

uploaded_file = st.file_uploader("Upload your image here", type=["jpg", "png", "jpeg"])
result = None
labels = None
probs = None

if uploaded_file is not None:

    st.image(uploaded_file, caption="Uploaded Image", width=150)

    if st.button("🔍 Classify Image"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
            status_text.text(f"Processing... {i+1}%")
        status_text.text("✅ Done!")

        result, (labels, probs) = prediction(uploaded_file)
        st.success("🎯 Prediction Result")
        st.markdown(f"**{result}**")

    st.markdown("---")

    if labels is not None and probs is not None:

        df = pd.DataFrame({
            "Label": labels,
            "Confidence": probs
        })

        figure = px.bar(
            df,
            y="Label",
            x="Confidence",
            orientation='h',
            color="Label",
            text="Confidence",
            title="Confidence Scores by Class",
            color_discrete_sequence=px.colors.qualitative.Set2
        )

        figure.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        figure.update_layout(
            xaxis=dict(range=[0, 1]),
            xaxis_title="Confidence Score",
            yaxis_title=None,
            showlegend=False,
            height=350
        )

        st.plotly_chart(figure, use_container_width=True)

st.markdown("---")

