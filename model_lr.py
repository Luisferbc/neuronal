from sklearn.linear_model import LinearRegression
import numpy as np

def train_lr():
    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]).reshape(-1, 1)
    ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0])

    model = LinearRegression()
    model.fit(xs, ys)

    return model
