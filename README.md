# 🧠 Streamlit Image Classification App

This is a user-friendly, web-based image classification application built using **Streamlit**. It allows users to:

* Upload zipped image datasets (organized by class folders)
* Choose test/train split
* Preprocess images
* Extract features (using **Gabor** and **Sobel** filters)
* Train a **Random Forest** model
* Evaluate using test data (accuracy & confusion matrix)
* Classify new uploaded images using the trained model

---

## 🚀 Live App

👉 [Click here to access the app](https://image-classification-project.streamlit.app/)

---

<img width="1536" height="1024" alt="ChatGPT Image Jul 23, 2025, 02_26_37 PM" src="https://github.com/user-attachments/assets/299438b0-bd2f-45f6-9e63-794f962c12db" />


## 📁 Features

* 🔄 Dynamic preprocessing
* 🔍 Feature extraction using Gabor + Sobel filters
* 🌲 Model training with Random Forest (n\_estimators = 70)
* 📊 Accuracy evaluation and confusion matrix visualization
* 🖼️ Upload your own image to test the trained model
* 🧠 Option to train your custom model with your dataset

---

## 🧪 Sample Dataset Structure

```
zip_file/
├── Class1/
│   ├── image1.jpg
│   ├── image2.jpg
├── Class2/
│   ├── image3.jpg
│   └── image4.jpg
```

---

## ⚙️ Tech Stack

* Python
* Streamlit
* Scikit-learn
* OpenCV
* Pandas / NumPy
* Matplotlib / Seaborn
* Joblib

---

## 📦 Setup Instructions

```bash
# 1. Clone the repo
https://github.com/YOUR_GITHUB_USERNAME/Streamlit_app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run locally
streamlit run app.py
```

---

## 🧠 Model Training Workflow

1. Upload zipped image dataset
2. Set test data split
3. Preprocess image data
4. Extract features
5. Train model
6. Evaluate model
7. Use classifier

---

## 📌 Notes

* Streamlit Cloud supports temporary model saving (during session)
* Works with `.jpg`, `.jpeg`, `.png` images
* Zipped dataset should contain folders (each representing a class)

---

## 🙌 Acknowledgements

* Built using Streamlit and Scikit-learn
* Designed for learning & demo purpose by \Taher Ali
