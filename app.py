import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from model_nn import train_nn
from model_lr import train_lr

st.title("🧠 Comparación: Red Neuronal vs Regresión Lineal")

# Sidebar controls
epochs = st.sidebar.slider("Épocas", 100, 1000, 500)
lr = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01)

# Train models
nn_model, history = train_nn(epochs, lr)
lr_model = train_lr()

# User input
x_input = st.number_input("Ingresa un valor de X", value=10.0)

# Predictions
nn_pred = nn_model.predict(np.array([[x_input]]))[0]
lr_pred = lr_model.predict([[x_input]])[0]

st.subheader("📌 Predicciones")
st.write(f"Red Neuronal: {nn_pred:.2f}")
st.write(f"Regresión Lineal: {lr_pred:.2f}")

# Data
xs = np.array([-1, 0, 1, 2, 3, 4])
ys = np.array([-2, 1, 4, 7, 10, 13])

x_range = np.linspace(-2, 12, 100)

nn_preds = nn_model.predict(x_range.reshape(-1, 1))
lr_preds = lr_model.predict(x_range.reshape(-1, 1))

# Plot
fig, ax = plt.subplots()
ax.scatter(xs, ys)
ax.plot(x_range, nn_preds, label="Red Neuronal")
ax.plot(x_range, lr_preds, linestyle="dashed", label="Regresión Lineal")
ax.legend()

st.pyplot(fig)

# Loss chart
st.subheader("📉 Pérdida del modelo NN")
st.line_chart(history["loss"])
