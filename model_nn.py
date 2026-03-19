from sklearn.neural_network import MLPRegressor
import numpy as np

def train_nn(epochs=500, lr=0.01):
    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]).reshape(-1, 1)
    ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0])

    model = MLPRegressor(
        hidden_layer_sizes=(10,),
        max_iter=epochs,
        learning_rate_init=lr,
        random_state=42
    )

    model.fit(xs, ys)

    history = {"loss": model.loss_curve_}

    return model, history
