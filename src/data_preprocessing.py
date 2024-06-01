import pandas as pd
import glob


def load_data(path_pattern):
    files = glob.glob(path_pattern)
    data_frames = [pd.read_excel(file) for file in files]
    data = pd.concat(data_frames)
    return data


def preprocess_data(data):
    # Select only numeric columns for normalization
    numeric_data = data.select_dtypes(include=[float, int])
    data_normalized = (
        numeric_data - numeric_data.mean()
    ) / numeric_data.std()  # Normalize the data
    return data_normalized


if __name__ == "__main__":
    # Load data from the 'data' directory
    data = load_data("data/*.xlsx")
    # Preprocess the loaded data
    data_normalized = preprocess_data(data)
    # Save the processed data to a new CSV file
    data_normalized.to_csv("processed_data.csv", index=False)
    print("Data preprocessing complete. Processed data saved to 'processed_data.csv'.")
