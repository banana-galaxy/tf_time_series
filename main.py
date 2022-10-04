import tensorflow as tf
import numpy as np
import pandas as pd
from random import randint
from tensorflow import keras
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os

# using a class for readability while keeping functionality
class Marker_predictor:
    def __init__(self):
        # get all the available market data for training and prediction
        data = pd.read_csv('EURUSD_Candlestick_1_Hour_BID_01.01.2010-02.05.2020.csv')
        self.close_data = data['Close'].tolist()

    def get_train_data(self, data, lags, amount):
        x = []
        y = []
        for i in range(amount):
            '''x.append([data[i]])
            delta = data[i + 1] - data[i]
            if -0.0001 <= delta <= 0.0001:
                y.append([0, 1, 0])
            elif data[i + 1] > data[i]:
                y.append([1, 0, 0])
            elif data[i + 1] < data[i]:
                y.append([0, 0, 1])'''
            x.append([])
            for u in range(lags):
                x[-1].append(data[i + u])
                print(data[i + u])
            delta = data[i + lags] - data[i + lags - 1]
            if -0.0001 <= delta <= 0.0001:
                y.append([0, 1, 0])
            elif data[i + lags] > data[i + lags - 1]:
                y.append([0, 0, 1])
            elif data[i + lags] < data[i + lags - 1]:
                y.append([1, 0, 0])
            # else:
            #    y.append([0,1,0])
        return x, y

    def train(self):
        x_train, y_train = self.get_train_data(self.close_data, 24, 40000)

        x = np.array(x_train)
        y = np.array(y_train)
        x = np.expand_dims(x, 1)


        model = tf.keras.Sequential()

        model.add(tf.keras.layers.LSTM(128, input_shape=(x.shape[1:]), activation='relu', return_sequences=True))
        model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.LSTM(128, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.1))

        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.Dense(3, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.fit(x, y, epochs=50)
        model.save('test_model')

    def predict(self):
        model = tf.keras.models.load_model('test_model')
        #print(model.predict(np.expand_dims(np.array([[4,5,6]]), 1)))

        prediction_data = []
        for i in range(41000, 42000):
            data = np.expand_dims(np.array([self.close_data[i]]), 1)
            sys.stdout = open(os.devnull, 'w')
            prediction = model.predict(data)[0]
            sys.stdout = sys.__stdout__
            print(prediction)
            prediction_data.append(prediction)
        plt.plot([i for i in range(1000)], self.close_data[41000:42000])
        plt.plot([i for i in range(1000)], prediction_data)
        plt.show()


neural_net = Marker_predictor()

neural_net.train()
neural_net.predict()