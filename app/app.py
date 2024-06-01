import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from predict import predict, LSTMModel

# Load data
data = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "processed_data.csv"))

# Load model
model = LSTMModel(input_size=1, hidden_layer_size=100, output_size=1)
model.load_state_dict(
    torch.load(os.path.join(os.path.dirname(__file__), "..", "lstm_model.pth"))
)
model.eval()

# Make predictions
test_inputs = torch.FloatTensor(data.values[-4:])  # Ensure test_inputs is a tensor
predicted_values = []
for _ in range(10):
    seq = test_inputs[-4:]
    predicted_value = predict(model, seq)
    predicted_values.append(predicted_value)
    test_inputs = torch.cat((test_inputs, torch.tensor([[predicted_value]])), 0)

# Display results
st.title("Time Series Forecasting")
st.line_chart(data)
st.write("Predicted Values:")
st.write(predicted_values)

# Plot the predicted values
fig, ax = plt.subplots()
ax.plot(range(len(data)), data, label="Actual")
ax.plot(
    range(len(data), len(data) + len(predicted_values)),
    predicted_values,
    label="Predicted",
)
ax.legend()
st.pyplot(fig)
