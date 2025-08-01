# utils/data_loader.py
import pandas as pd

def load_dataset(path="data/track2_train/track2_simple.csv"):
    df = pd.read_csv(path)
    return df['content'].tolist(), df.iloc[:, 2:].values
