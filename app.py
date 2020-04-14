import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
import os
import csv
import json
import pickle
import keras
import warnings
import tensorflow as tf
import xgboost

from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import load_model, Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.backend import tensorflow_backend as tb

warnings.filterwarnings('ignore')
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
app = Flask(__name__)
graph = tf.get_default_graph()

@app.route('/')
def landing_point():
    return 'THIS IS THE LANDING PAGE. PLEASE USE "/predict"'


@app.route('/predict', methods=['POST'])
def predict():
    def jsonrequest_to_dataframe(json_data):
        final_data = []
        for sample_indx in range(len(json_data)):
            sample = []
            for body_part in json_data[sample_indx]['keypoints']:
                sample.append(body_part['position']['x'])
                sample.append(body_part['position']['y'])
            final_data.append(sample)
        final_data = np.array(final_data)

        columns = ['nose_x', 'nose_y',
                   'leftEye_x', 'leftEye_y', 'rightEye_x', 'rightEye_y',
                   'leftEar_x', 'leftEar_y', 'rightEar_x', 'rightEar_y',
                   'leftShoulder_x', 'leftShoulder_y', 'rightShoulder_x', 'rightShoulder_y',
                   'leftElbow_x', 'leftElbow_y', 'rightElbow_x', 'rightElbow_y',
                   'leftWrist_x', 'leftWrist_y', 'rightWrist_x', 'rightWrist_y',
                   'leftHip_x', 'leftHip_y', 'rightHip_x', 'rightHip_y',
                   'leftKnee_x', 'leftKnee_y', 'rightKnee_x', 'rightKnee_y',
                   'leftAnkle_x', 'leftAnkle_y', 'rightAnkle_x', 'rightAnkle_y']
        df = pd.DataFrame(final_data, columns=columns)
        df.to_csv('data.csv', index=False)
        return df

    def normalize_sample(X):
        x_num = X[:, 0].copy()
        y_num = X[:, 1].copy()

        total_cols = X.shape[-1]
        for col in range(0, total_cols, 2):
            X[:, col] = X[:, col] - x_num

        for col in range(1, total_cols, 2):
            X[:, col] = X[:, col] - y_num

        return X[:, 2:]

    def extend_data(X, diff, kind="zeros"):
        def zeros(sample, diff, num_features):
            return np.full(shape=(diff, num_features), fill_value=0)

        def means(sample, diff, num_features):
            mean_array = np.reshape(np.mean(sample, axis=0), (1, num_features))
            return np.repeat(mean_array, diff, axis=0)

        def copies(sample, diff, num_features):
            last_array = np.reshape(sample[-1], (1, num_features))
            return np.repeat(last_array, diff, axis=0)

        num_features = X.shape[-1]
        switcher = {"zeros": zeros(X, diff, num_features),
                    "means": means(X, diff, num_features),
                    "copies": copies(X, diff, num_features)}

        append_array = switcher[kind]
        X = np.append(X, append_array, axis=0)

        return X

    def feature_extraction(df):
        feature_list = ["nose_x", "nose_y", "leftShoulder_x", "leftShoulder_y", "rightShoulder_x", "rightShoulder_y",
                        "leftElbow_x", "leftElbow_y", "rightElbow_x", "rightElbow_y", "leftWrist_x", "leftWrist_y",
                        "rightWrist_x", "rightWrist_y"]
        max_timeframes = 250
        num_features = len(feature_list) - 2

        # Extract useful columns only
        data = df[feature_list].to_numpy()

        # Normalize data
        data = normalize_sample(data)

        # Equalize the timeframes
        diff = max_timeframes - data.shape[0]
        if (diff > 0):
            data = extend_data(data, diff, kind='means')
        else:
            data = data[:max_timeframes, :]

        # Scaling data
        data = data.reshape((-1, max_timeframes * num_features))
        scaler = pickle.load(open(os.path.join('IPD', 'scaler.pkl'), 'rb'))
        data = scaler.transform(data)
        data = data.reshape((max_timeframes, num_features))

        return data

    def test_models(X):
        num_timeframes, num_features = X.shape

        # Reshape data for models
        data_2dfeatures = X.reshape((1, num_timeframes, num_features, 1))
        data_1dfeatures = X.reshape((1, num_timeframes * num_features))

        model1_pred = model_1.predict(data_1dfeatures)[0]
        model2_pred = model_2.predict(data_1dfeatures)[0]
        model3_pred = model_4.predict(data_1dfeatures)[0]

        global graph
        with graph.as_default():
            model4_pred = model_4.predict_classes(data_2dfeatures)[0]
        tf.keras.backend.clear_session()
        
        label_dict = {'0':'buy','1':'communicate','2':'fun','3':'hope','4':'mother','5':'really'}

        return {"1": label_dict[str(model1_pred)], "2": label_dict[str(model2_pred)], 
                "3": label_dict[str(model3_pred)], "4": label_dict[str(model4_pred)]}

    # with open(os.path.join('data','json','buy','BUY_1_BAKRE.json')) as file:
    #     json_data = json.load(file)

    json_data = request.get_json()
    data = jsonrequest_to_dataframe(json_data)
    data = feature_extraction(data)
    pred_dict = test_models(data)

    return jsonify(pred_dict)


if __name__ == '__main__':
    # Loading trained models
    model_1 = pickle.load(open(os.path.join("models", "SVM_model.pkl"), 'rb'))
    model_2 = pickle.load(open(os.path.join("models", "RandomForest_model.pkl"), 'rb'))
    model_3 = pickle.load(open(os.path.join("models", "KNN_model.pkl"), 'rb'))

    model_4 = Sequential()
    model_4.add(Conv2D(6, kernel_size=(5, 5),activation='tanh',
                 input_shape=(250,12,1)))
    model_4.add(MaxPooling2D(pool_size=(2, 1)))
    model_4.add(Conv2D(12, (3, 3), activation='tanh'))
    model_4.add(MaxPooling2D(pool_size=(2, 1)))
    model_4.add(Dropout(0.25))
    model_4.add(Flatten())
    model_4.add(Dense(12, activation='tanh'))
    model_4.add(Dropout(0.25))
    model_4.add(Dense(6, activation='softmax'))

    weights = pickle.load(open(os.path.join("models", "CNN_model.pkl"), "rb"))
    model_4.set_weights(weights)
    model_4.compile(loss=keras.losses.categorical_crossentropy,
                   optimizer=keras.optimizers.Adadelta(),
                   metrics=['accuracy'])

    app.run(host='127.0.0.1', port=8080, debug=True)