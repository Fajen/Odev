import tensorflow as tf
import numpy as np

class TFQuadraticModel:
    def __init__(self):
        self.model = tf.keras.Sequential([
            tf.keras.Input(shape=(1,)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1)
        ])

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            loss="mse"
        )

    def train(self, x, y, epochs=500):
        self.model.fit(x, y, epochs=epochs, verbose=0)

    def predict(self, x):
        x = np.array(x, dtype=float)
        return self.model.predict(x, verbose=0)



print("TensorFlow Quadratic Model")

x_train = np.linspace(-10, 10, 100)
y_train = x_train**2 + 2*x_train + 1

tf_model = TFQuadraticModel()
tf_model.train(x_train, y_train)

test_x = [-3, 0, 2, 5]
predictions = tf_model.predict(test_x)

for x, y in zip(test_x, predictions):
    print(f"x={x} -> y≈{y[0]:.2f}")
