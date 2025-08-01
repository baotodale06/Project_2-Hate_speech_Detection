# utils/data_loader.py
import pandas as pd

def load_dataset(path="data/track2_train/track2_simple.csv"):
    df = pd.read_csv(path)
    ## delete "Not Related" class
    return df['content'].tolist(), df.iloc[:, 2:7].values
