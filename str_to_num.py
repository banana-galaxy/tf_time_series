import numpy as np
import tensorflow as tf

x_train = np.array(['1','0','1','1','0'])
y_train = np.array([1,0,1,1,0])
x_train = np.expand_dims(x_train, 1)
print(x_train.shape)

model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(64, input_shape=(5, 1)))
model.add(tf.keras.layers.Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(x_train, y_train, epochs=100)