import numpy as np
import glob
import cv2
import os
import pandas as pd
from skimage.filters import sobel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from model_function import feature_extractor
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

def image_uploading(tmp_dir):
    train_images = []
    train_labels = []
    SIZE = 128
    nested_folder = [f for f in glob.glob(os.path.join(tmp_dir, "*")) if os.path.isdir(f)][0]

    for directory_path in glob.glob(os.path.join(nested_folder, "*")):
        label = os.path.basename(directory_path)
        print(label)
        count = 0
        images_path = (
                glob.glob(os.path.join(directory_path, "*.png")) +
                glob.glob(os.path.join(directory_path, "*.jpeg")) +
                glob.glob(os.path.join(directory_path, "*.jpg"))
        )

        for img_path in images_path:
            print(img_path)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.resize(img, (SIZE, SIZE))
                train_images.append(img)
                train_labels.append(label)
                count += 1
            if count == 600:
                break

    return train_images, train_labels


def image_preprocessing(train_images, train_labels, size):
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    # Splitting data into test and train
    x_train, x_test, y_train, y_test = train_test_split(
        train_images, train_labels,
        test_size=size,  # 20% for testing
        stratify=train_labels,
        random_state=42
    )

    # Label Encoding
    le = LabelEncoder()
    le.fit(y_train)

    y_train_encoded = le.transform(y_train)
    y_test_encoded = le.transform(y_test)

    y_train = y_train_encoded
    y_test = y_test_encoded

    # y_test_decoder = le.inverse_transform(y_test)
    # print(y_test_decoder)
    # print(y_test)

    # Normalize Pixel values to between 0 and 1
    x_train, x_test = x_train/255.0, x_test/255.

    return x_train, x_test, y_train, y_test, le

def features(x_train):
    image_features = feature_extractor(x_train)
    return image_features


def train_model(image_features, x_train, y_train):

    X_for_RF = np.reshape(image_features, (x_train.shape[0], -1))
    RF_model = RandomForestClassifier(n_estimators=50, random_state=42)

    RF_model.fit(X_for_RF, y_train)

    return RF_model, 1


def test_prediction(RF_model, x_test, y_test):
    test_features = feature_extractor(x_test)
    test_for_RF = np.reshape(test_features, (x_test.shape[0], -1))

    prediction = RF_model.predict(test_for_RF)

    return metrics.accuracy_score(y_test, prediction), prediction



