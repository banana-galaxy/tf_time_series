import tensorflow as tf
import numpy as np
from random import randint

def gen_data(total, lags):
    x = []
    y = []
    for i in range(total):
        num = randint(0, 100)
        x.append([])
        for u in range(lags):
            x[-1].append(num+u)
        #y.append(num+lags)  # append the next number in the sequence to train the net for sequence prediction
        y.append(sum([num+i for i in range(lags)]))   # append the sum of the sequence to train the net to add the 3 numbers together
    return x, y

x, y = gen_data(5000, 3)

x_train = np.array(x)
y_train = np.array(y)
x_train = np.expand_dims(x_train, 1)
print(x_train.shape)

model = tf.keras.Sequential()

model.add(tf.keras.layers.LSTM(100, return_sequences=True, input_shape=x_train.shape[1:]))
model.add(tf.keras.layers.LSTM(100))
model.add(tf.keras.layers.Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(x_train, y_train, epochs=40)

x_predict = np.expand_dims(np.array([[1,2,3]]), 1)

print(model.predict(x_predict))