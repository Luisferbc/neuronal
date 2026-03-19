import tensorflow as tf
import numpy as np

def train_nn(epochs=500, lr=0.01):
    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=1, input_shape=[1])
    ])

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
        loss='mean_squared_error'
    )

    history = model.fit(xs, ys, epochs=epochs, verbose=0)

    return model, history
