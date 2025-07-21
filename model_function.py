import numpy as np
import pandas as pd
import cv2
from skimage.filters import sobel
import joblib
from sklearn import metrics

RF_model = joblib.load('my_model.pkl')

def feature_extractor(dataset):
    x_train = dataset
    image_dataset = pd.DataFrame()

    for image in range(x_train.shape[0]):

        df = pd.DataFrame()

        input_img = x_train[image, :, :, :]
        img = input_img

        # FEATURE 1 - Pixel values

        pixel_values = img.reshape(-1)
        df['Pixel_Value'] = pixel_values

        # G

        num = 1
        kernels = []
        for theta in range(2):
            theta = theta / 4. * np.pi
            for sigma in range(1,3):
                lamda = np.pi/4
                gamma = 0.5
                gabor_label = "Gabor" + str(num)

                ksize = 9
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
                kernels.append(kernel)

                #Now filter the image and add values to a new column
                fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                filtered_img = fimg.reshape(-1)
                df[gabor_label] =  filtered_img
                print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                num += 1

        edge_sobel = sobel(img)
        edge_sobel1 = edge_sobel.reshape(-1)
        df["Sobel"] = edge_sobel1

        image_dataset = pd.concat([image_dataset, df], ignore_index=True)

    return image_dataset


def prediction(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (128, 128))
    img = img/255.0

    input_img = np.expand_dims(img, axis=0)
    r_image_features = feature_extractor(input_img)
    print(r_image_features.head())
    r_image_features = np.expand_dims(r_image_features, axis=0)
    r_for_RF = np.reshape(r_image_features, (1, -1))
    image_prediction = RF_model.predict(r_for_RF)

    # probability
    probs = RF_model.predict_proba(r_for_RF)[0]
    labels = ["Cat", "Dog"]
    for label, prob in zip(labels ,probs):
        print(f"{label}: {prob:.2f}")

    return labels[int(image_prediction)], (labels, probs)

