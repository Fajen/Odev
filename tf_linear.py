import tensorflow as tf
import numpy as np

x = np.array([0, 1, 2, 3, 4], dtype=float)
y = np.array([1, 3, 5, 7, 9], dtype=float)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

model.compile(
    optimizer='sgd',
    loss='mean_squared_error'
)

model.fit(x, y, epochs=300, verbose=0)

test_x = np.array([10.0])
prediction = model.predict(test_x, verbose=0)

print("TensorFlow Prediction for x=10:", prediction[0][0])
