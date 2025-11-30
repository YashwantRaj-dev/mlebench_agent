import os
import pandas as pd
from PIL import Image

def detect_dataset_type(data_path):
    files = os.listdir(data_path)

    if any(f.endswith('.jpg') or f.endswith('.png') for f in files):
        return "image"
    if any("text" in f.lower() for f in files) or any(f.endswith('.txt') for f in files):
        return "text"
    if any(f.endswith('.csv') for f in files):
        df = pd.read_csv(os.path.join(data_path, [f for f in files if f.endswith('.csv')][0]))
        # If text column with multiple words → NLP
        if any(df[col].dtype == object for col in df.columns):
            long_cols = [col for col in df.columns if df[col].astype(str).str.split().str.len().mean() > 3]
            return "text" if long_cols else "tabular"
        return "tabular"

    return None
