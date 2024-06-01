import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Ścieżka do katalogu z plikami
data_dir = "C:/Workspaces/AgriPricePredictor/data/"

files = [
    os.path.join(data_dir, "14_tydz_2021.xlsx"),
    os.path.join(data_dir, "17_tydz_2022.xlsx"),
    os.path.join(data_dir, "26_tydz_2023.xlsx"),
    os.path.join(data_dir, "27_tydz_2023.xlsx"),
    os.path.join(data_dir, "28_tydz_2023.xlsx"),
    os.path.join(data_dir, "29_tydz_2023.xlsx"),
    os.path.join(data_dir, "30_tydz_2023.xlsx"),
]


# Funkcja do ekstrakcji daty i ceny z pliku
def extract_date_and_price(file):
    df = pd.read_excel(file)
    date_column = df.columns[1]
    date_str = date_column.split()[-1]
    date = pd.to_datetime(date_str)
    price_column = date_column
    prices = df[price_column]
    return pd.DataFrame({"date": date, "price": prices})


# Połączenie danych z wszystkich plików
all_data = pd.concat([extract_date_and_price(file) for file in files])

# Usunięcie wierszy z brakującymi danymi
all_data.dropna(inplace=True)

# Ustawienie indeksu i sortowanie
all_data.set_index("date", inplace=True)
all_data.sort_index(inplace=True)

# Normalizacja danych
scaler = MinMaxScaler(feature_range=(-1, 1))
all_data["price"] = scaler.fit_transform(all_data[["price"]])


# Przygotowanie danych do modelu
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i : i + seq_length]
        label = data[i + seq_length]
        sequences.append((seq, label))
    return sequences


seq_length = 4
sequences = create_sequences(all_data["price"].values, seq_length)

# Podział na zbiór treningowy i testowy
train_size = int(len(sequences) * 0.8)
train_sequences = sequences[:train_size]
test_sequences = sequences[train_size:]


# Konwersja danych na tensory
def to_tensors(sequences):
    X = np.array([seq for seq, label in sequences])
    y = np.array([label for seq, label in sequences])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


X_train, y_train = to_tensors(train_sequences)
X_test, y_test = to_tensors(test_sequences)


# Definicja modelu
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super(LSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1])
        return predictions


model = LSTM()
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Trenowanie modelu
epochs = 300  # zwiększenie liczby epok

for epoch in range(epochs):
    for seq, labels in train_sequences:
        optimizer.zero_grad()

        seq_tensor = torch.tensor(seq, dtype=torch.float32).view(
            1, -1, 1
        )  # przygotowanie seq do modelu
        y_pred = model(seq_tensor)

        single_loss = loss_function(
            y_pred, torch.tensor(labels, dtype=torch.float32).view(1, -1)
        )
        single_loss.backward()
        optimizer.step()

    if epoch % 25 == 0:
        print(f"Epoch {epoch} loss: {single_loss.item()}")

# Ocena modelu
model.eval()
test_predictions = []

for seq, _ in test_sequences:
    with torch.no_grad():
        seq_tensor = torch.tensor(seq, dtype=torch.float32).view(
            1, -1, 1
        )  # przygotowanie seq do modelu
        test_predictions.append(model(seq_tensor).item())

# Denormalizacja wyników
test_predictions = scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1))
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Wizualizacja wyników
plt.figure(figsize=(10, 6))
plt.plot(all_data.index[-len(test_predictions) :], test_predictions, label="Predicted")
plt.plot(all_data.index[-len(test_predictions) :], y_test, label="Actual")
plt.legend()
plt.show()
