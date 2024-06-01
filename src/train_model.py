import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from lstm_model import LSTMModel
from utils import create_inout_sequences

# Load and preprocess data
data = pd.read_csv("processed_data.csv")

# Prepare training data
train_window = 4
train_inout_seq = create_inout_sequences(data.values, train_window)

# Define model, loss function, and optimizer
model = LSTMModel(input_size=1, hidden_layer_size=100, output_size=1)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 150
for i in range(epochs):
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        model.hidden_cell = (
            torch.zeros(1, 1, model.hidden_layer_size),
            torch.zeros(1, 1, model.hidden_layer_size),
        )
        y_pred = model(torch.FloatTensor(seq))
        single_loss = loss_function(y_pred, torch.FloatTensor(labels))
        single_loss.backward()
        optimizer.step()
    if i % 25 == 1:
        print(f"epoch: {i:3} loss: {single_loss.item():10.8f}")

# Save the trained model
torch.save(model.state_dict(), "lstm_model.pth")
print("Model training complete. Model saved to 'lstm_model.pth'.")
