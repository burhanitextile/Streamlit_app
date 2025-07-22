
import streamlit as st
from model_function import prediction
from training_model import image_uploading, image_preprocessing, features, train_model, test_prediction
import time
import pandas as pd
import plotly.express as px
import zipfile
import tempfile
import os
import joblib
from sklearn import metrics
from streamlit_navigation_bar import st_navbar  # or the community version
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Define pages
pages = ["Image Classifier", "Model Trainer", "About"]
current = st_navbar(pages)

# Render content based on selection
if current == "Image Classifier":


    st.set_page_config(page_title="Image Classification", layout="centered")

    st.title("Image Classification App")
    st.write("Upload an image of animal (🐱Cat and 🐶Dog) and let the model predict what it is!")

    # Initialize photo state
    if "photo" not in st.session_state:
        st.session_state["photo"] = "not done"


    def change_photo_state():
        st.session_state["photo"] = "done"


    # with st.sidebar:
    #     st.header("Instructions")
    #     st.markdown("""
    #     - Upload an image file (e.g., JPG, PNG).
    #     - Click the 'Classify Image' button.
    #     - Wait for the model to process and return the result.
    #     """)

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
                status_text.text(f"Processing... {i + 1}%")
            status_text.text("✅ Done!")

            if os.path.exists("user_model.pkl") and os.path.exists("lebal_encoder.pkl"):
                new_model = joblib.load("user_model.pkl")
                le = joblib.load("lebal_encoder.pkl")
                result, (labels, probs) = prediction(uploaded_file, new_model, le)
            else:
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
elif current == "Model Trainer":
    st.title("🧠 Model Trainer")

    st.markdown("""
        Train your own machine learning model by uploading a dataset of labeled images.
        This tool allows you to customize the train-test split, preprocess images, extract features,
        and evaluate the model — all in one place.
        """)

    # Upload ZIP
    st.markdown("#### 📦 Upload Dataset (ZIP File)")
    uploaded_zip = st.file_uploader("Upload a Zip file", type="zip")

    if uploaded_zip and st.button("📤 Extract and Load Images"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
                zip_ref.extractall(tmp_dir)

            train_images, train_labels = image_uploading(tmp_dir)
            st.session_state.train_images = train_images
            st.session_state.train_labels = train_labels
            st.success("✅ Images loaded and stored!")

    st.divider()

    st.markdown("#### 🎚️ Set Test Data Percentage")
    test_size = st.slider(
        "Select Test Data Percentage",
        min_value=10,
        max_value=50,
        value=20,
        step=5,
        format="%d%%",
        width=250
    )
    test_ratio = test_size / 100

    st.divider()

    st.markdown("#### 🛠️ Image Preprocessing ")
    st.markdown("""
    This step prepares the image data for training by:
    - Splitting it into training and testing sets using the selected test size
    - Encoding class labels into numeric form using `LabelEncoder`
    - Normalizing pixel values to a 0–1 range for better model performance

    These preprocessing steps ensure your model is trained on clean, consistent, and well-structured data.
    """)
    if st.button("🛠️ Process Dataset"):
        if "train_images" in st.session_state:
            x_train, x_test, y_train, y_test, le = image_preprocessing(
                st.session_state.train_images,
                st.session_state.train_labels,
                test_ratio
            )
            st.session_state.x_train = x_train
            st.session_state.x_test = x_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.le = le
            st.success("✅ Data processed and saved!")
        else:
            st.warning("⚠️ Please upload and load images first.")
    st.divider()

    st.markdown("#### 🔍 Feature Extraction")
    st.markdown("""
    This step extracts meaningful features from the preprocessed images to help the model learn better.
    By default, it uses a combination of the `Gabor` filter and `Sobel` edge detection to capture texture and edge-based patterns.
    These features improve the model's ability to distinguish between different image classes.
    """)
    if st.button("🔍 Extract Features"):
        st.caption("⚠️ Feature extraction may take a few moments. Please wait...")
        if "x_train" in st.session_state:
            image_features = features(st.session_state.x_train)
            st.session_state.image_features = image_features
            st.success("✅ Features extracted and saved!")
        else:
            st.warning("⚠️ Please preprocess the data first.")

    st.divider()

    st.markdown("#### 📊 Model Training and Evaluation ")
    st.markdown("""
    In this step, a machine learning model is trained using the extracted features and their corresponding labels.
    We use a `Random Forest Classifier` with `n_estimators = 70`, which means the model builds an ensemble of 70 decision trees.

    This ensemble approach helps improve prediction accuracy and reduces overfitting by combining the strengths of multiple trees.
    """)
    if st.button("📊 Train Model"):
        if "image_features" in st.session_state:
            new_model, flag = train_model(
                st.session_state.image_features,
                st.session_state.x_train,
                st.session_state.y_train
            )
            st.session_state.model = new_model
            joblib.dump(new_model, "user_model.pkl")
            joblib.dump(st.session_state.le, "lebal_encoder.pkl")
            st.success("✅ Model trained and saved!")
        else:
            st.warning("⚠️ Please extract features first.")
    st.divider()

    st.markdown("#### 📈 Evaluate Model")
    st.markdown("""
    This section evaluates the trained model using the test dataset that was set aside earlier.

    It calculates the `overall accuracy` and provides visual insights such as the `confusion matrix` to show how well the model distinguishes between classes.

    These metrics help you understand the model's performance and identify any weaknesses in predictions.
    """)
    if st.button("📈 Evaluate on Test Set"):
        if "model" in st.session_state:
            accuracy, y_pred = test_prediction(
                st.session_state.model,
                st.session_state.x_test,
                st.session_state.y_test
            )
            st.session_state.y_pred = y_pred
            st.success(f"✅ Model Accuracy: **{accuracy:.2f}%**")
        else:
            st.warning("⚠️ Please train the model first.")
    if st.button("📊 Show Confusion Matrix"):
        if "model" in st.session_state:
            # Make predictions
            y_pred = st.session_state.y_pred
            y_true = st.session_state.y_test

            # Generate confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            labels = st.session_state.le.classes_

            # Plot using seaborn
            fig, ax = plt.subplots(figsize=(6, 3))
            sns.heatmap(cm, annot=True, fmt='d',  xticklabels=labels, yticklabels=labels, ax=ax)  # cmap='Blues',
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")

            st.pyplot(fig)
        else:
            st.warning("⚠️ Please train the model first.")

        st.markdown("---")
        st.markdown(
            "✅ **Your model is ready!** To test it on your own images, head over to the **📷 Image Classifier** page.")


else:
    st.write("ℹ️ About page")

