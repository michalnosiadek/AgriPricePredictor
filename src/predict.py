import torch
import pandas as pd
from lstm_model import LSTMModel


def predict(model, input_seq):
    model.eval()
    with torch.no_grad():
        model.hidden_cell = (
            torch.zeros(1, 1, model.hidden_layer_size),
            torch.zeros(1, 1, model.hidden_layer_size),
        )
        return model(torch.FloatTensor(input_seq)).item()


# Load the model
model = LSTMModel(input_size=1, hidden_layer_size=100, output_size=1)
model.load_state_dict(torch.load("lstm_model.pth"))
model.eval()

# Load and preprocess data
data = pd.read_csv("processed_data.csv")
test_inputs = data.values[-4:]  # Last 4 values as input for prediction

# Make predictions
predicted_values = []
for _ in range(10):  # Predict next 10 values
    seq = test_inputs[-4:]
    predicted_value = predict(model, seq)
    predicted_values.append(predicted_value)
    test_inputs = torch.cat((test_inputs, torch.tensor([[predicted_value]])), 0)

print("Predicted values:", predicted_values)
