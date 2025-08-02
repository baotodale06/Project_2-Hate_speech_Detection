# utils/data_loader.py
import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(path="data/track2_train/track2_simple.csv"):
    df = pd.read_csv(path)
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    ## delete "Not Related" class
    return df_train['content'].tolist(), df_train.iloc[:, 2:7].values, df_test['content'].tolist(), df_test.iloc[:, 2:7].values
