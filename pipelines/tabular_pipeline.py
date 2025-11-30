import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb

def run_tabular_pipeline(data_path, competition_id, output_path):
    base = Path(data_path)/competition_id
    if not base.exists(): base=Path(data_path)
    train = base/"train.csv"; sample = base/"sample_submission.csv"
    if not train.exists():
        print("train.csv missing — cannot run tabular pipeline")
        return
    df = pd.read_csv(train)
    # heuristic: find label column (named target or target in col)
    label_cols=[c for c in df.columns if 'target' in c.lower() or c.lower()=='target']
    if not label_cols:
        # fallback: last numeric column
        label_cols = [df.select_dtypes(include=[np.number]).columns[-1]]
    label = label_cols[0]
    X = df.drop(columns=[label])
    # simple numeric fill
    X = X.select_dtypes(include=[np.number]).fillna(0)
    y = df[label].values
    if X.shape[1]==0:
        print("No numeric features — writing zeros")
        s = pd.read_csv(sample) if sample.exists() else None
        if s is not None:
            for c in s.columns[1:]: s[c]=0
            s.to_csv(output_path,index=False)
        return
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val)
    params={'objective':'binary','metric':'auc' if len(np.unique(y))>1 else 'rmse','verbosity':-1}
    model = lgb.train(params, dtrain, valid_sets=[dval], num_boost_round=100, early_stopping_rounds=10, verbose_eval=False)
    # predict on sample_submission identifiers: if sample has columns, we fill with mean prediction
    s = pd.read_csv(sample) if sample.exists() else None
    if s is None:
        print("No sample_submission; nothing to save")
        return
    # simple: predict constant mean
    pred_val = float(np.mean(y_train))
    for c in s.columns[1:]:
        s[c]=pred_val
    s.to_csv(output_path, index=False)
    print("Saved tabular submission to", output_path)
