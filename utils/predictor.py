import pandas as pd

def save_predictions(preds, out_path):
    sub = pd.DataFrame({"id": range(len(preds)), "label": preds})
    sub.to_csv(out_path, index=False)
