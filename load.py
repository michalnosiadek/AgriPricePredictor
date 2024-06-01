import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import os
import time

# Wczytanie danych
data = pd.read_csv("data/ProductPriceIndex.csv")

# Zmiana nazw kolumn na polskie
data.rename(
    columns={
        "productname": "nazwa_produktu",
        "date": "data",
        "farmprice": "cena_gospodarstwa",
        "atlantaretail": "cena_detaliczna_atlanta",
        "chicagoretail": "cena_detaliczna_chicago",
        "losangelesretail": "cena_detaliczna_los_angeles",
        "newyorkretail": "cena_detaliczna_nowy_jork",
        "averagespread": "srednia_marza",
    },
    inplace=True,
)

# Konwersja kolumny data na typ datetime
data["data"] = pd.to_datetime(data["data"])
data.set_index("data", inplace=True)

# Usunięcie znaków dolara i konwersja cen na typ numeryczny
for column in [
    "cena_gospodarstwa",
    "cena_detaliczna_atlanta",
    "cena_detaliczna_chicago",
    "cena_detaliczna_los_angeles",
    "cena_detaliczna_nowy_jork",
]:
    data[column] = (
        data[column].replace("[\$,]", "", regex=True).replace("", np.nan).astype(float)
    )

# Lista unikalnych produktów
products = data["nazwa_produktu"].unique()
print(f"Produkty: {products}")

# Upewnienie się, że folder forecast_data istnieje
if not os.path.exists("forecast_data"):
    os.makedirs("forecast_data")


# Definicja modelu LSTM
class LSTM(nn.Module):
    def __init__(
        self, input_size=1, hidden_layer_size=20, output_size=1
    ):  # Zmniejszenie liczby neuronów w warstwie ukrytej
        super(LSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (
            torch.zeros(1, 1, self.hidden_layer_size),
            torch.zeros(1, 1, self.hidden_layer_size),
        )

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(
            input_seq.view(len(input_seq), 1, -1), self.hidden_cell
        )
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


# Przygotowanie danych treningowych
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i : i + tw]
        train_label = input_data[i + tw : i + tw + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq


train_window = 12

# Iteracja po każdym produkcie
for product in products:
    print(f"\nPrzetwarzanie produktu: {product}")
    product_data = data[data["nazwa_produktu"] == product].copy()

    # Usunięcie wierszy z brakującymi wartościami
    product_data = product_data.dropna(subset=["cena_gospodarstwa"])

    if len(product_data) < train_window + 1:
        print(f"Za mało danych dla produktu {product}")
        continue

    print(f"Liczba wierszy dla produktu {product}: {len(product_data)}")

    # Skalowanie danych
    scaler = MinMaxScaler()
    product_data["cena_gospodarstwa"] = scaler.fit_transform(
        product_data[["cena_gospodarstwa"]]
    )

    # Podział danych na zestaw treningowy i testowy
    train_size = int(len(product_data) * 0.8)
    train_data = product_data.iloc[:train_size]
    test_data = product_data.iloc[train_size:]

    print(f"\nProdukt: {product}")
    print("Dane treningowe:")
    print(train_data.head())
    print("\nDane testowe:")
    print(test_data.head())

    train_inout_seq = create_inout_sequences(
        train_data["cena_gospodarstwa"].values, train_window
    )

    # Inicjalizacja modelu, funkcji kosztu i optymalizatora
    model = LSTM()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Trening modelu
    epochs = 5  # Zmniejszenie liczby epok
    for epoch in range(epochs):
        start_time = time.time()
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

        end_time = time.time()
        epoch_time = end_time - start_time
        print(
            f"epoch: {epoch+1} loss: {single_loss.item():10.8f} time: {epoch_time:.2f} sec"
        )

    # Prognozowanie przyszłych wartości
    model.eval()

    future_inputs = train_data["cena_gospodarstwa"][-train_window:].tolist()
    future_predictions = []

    for i in range(4):  # Prognozowanie na 4 tygodnie do przodu
        seq = torch.FloatTensor(future_inputs[-train_window:])
        with torch.no_grad():
            model.hidden_cell = (
                torch.zeros(1, 1, model.hidden_layer_size),
                torch.zeros(1, 1, model.hidden_layer_size),
            )  # Reset hidden states
            prediction = model(seq).item()
            future_predictions.append(prediction)
            future_inputs.append(prediction)

    future_predictions = scaler.inverse_transform(
        np.array(future_predictions).reshape(-1, 1)
    )

    # Przekształć prognozy do oryginalnej skali (dolary)
    future_predictions_dollars = [
        "${:.2f}".format(pred[0]) for pred in future_predictions
    ]

    # Daty dla prognoz
    last_date = product_data.index[-1]
    future_dates = pd.date_range(start=last_date, periods=5, freq="W")[
        1:
    ]  # Następne 4 tygodnie

    # Zapis wyników do pliku CSV
    forecast_df = pd.DataFrame(
        data={
            "data": future_dates,
            "prognozowana_cena_gospodarstwa": future_predictions_dollars,
        }
    )
    forecast_df.to_csv(f"forecast_data/forecast_{product}.csv", index=False)

    print(f"Prognozy zapisane do pliku: forecast_data/forecast_{product}.csv")
