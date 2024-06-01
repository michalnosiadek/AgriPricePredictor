# Prognozowanie cen produktów rolnych

## Opis projektu

Ten projekt prognozuje przyszłe ceny produktów rolnych na podstawie danych historycznych przy użyciu modelu LSTM zaimplementowanego w PyTorch. Dane wejściowe są w formacie CSV i prognozy są zapisywane w formacie CSV w folderze `forecast_data`.

## Struktura projektu

- `load.py` - Główny skrypt Pythona do wczytywania danych, trenowania modelu i prognozowania przyszłych cen.
- `data/` - Folder zawierający dane wejściowe (`ProductPriceIndex.csv`).
- `forecast_data/` - Folder, w którym są zapisywane prognozy w formacie CSV.
- `requirements.txt` - Lista wymaganych bibliotek Pythona.

## Instalacja

1. **Klonowanie repozytorium**:
   Skopiuj projekt na swój komputer:
   ```sh
   git clone <URL do twojego repozytorium>
   cd <nazwa-repozytorium>
   ```

## Uruchomienie skryptu

python load.py
