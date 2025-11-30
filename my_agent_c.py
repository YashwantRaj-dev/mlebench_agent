#!/usr/bin/env python3
import os,sys,traceback
from pathlib import Path

from pipelines.image_pipeline import run_image_pipeline
from pipelines.text_pipeline import run_text_pipeline
from pipelines.tabular_pipeline import run_tabular_pipeline
from pipelines.seq2seq_pipeline import run_seq2seq_pipeline

COMPETITION_ID = os.getenv("COMPETITION_ID")
DATA_PATH = os.getenv("DATA_PATH", "/home/data")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "/home/data/submission.csv")

def infer_task_type(comp_id: str, data_path: str):
    cid = (comp_id or "").lower()
    if any(x in cid for x in ["isic","melanoma","cactus","aerial","whale","icml","whale"]):
        return "image"
    if "spooky" in cid or "author" in cid or "toxic" in cid:
        return "text_classification"
    if "text-normalization" in cid or "normalization" in cid:
        return "seq2seq"
    if "tabular" in cid or "playground" in cid or "titanic" in cid:
        return "tabular"
    # fallback: detect by files in prepared public
    p = Path(data_path) / (comp_id or "")
    if not p.exists():
        p = Path(data_path)
    # check image
    for ext in ('.jpg','.jpeg','.png'):
        if any((p / fname).exists() for fname in os.listdir(str(p)) if fname.lower().endswith(ext)):
            return "image"
    # check train.csv + content
    train = p / "train.csv"
    if train.exists():
        import pandas as pd
        df = pd.read_csv(train, nrows=5)
        cols = [c.lower() for c in df.columns]
        if any("text" in c or "sentence" in c for c in cols):
            return "text_classification"
        if any("target" in c for c in cols) or df.select_dtypes(include=['number']).shape[1] > 2:
            return "tabular"
    return "unknown"

def main():
    if COMPETITION_ID is None:
        print("ERROR: COMPETITION_ID not set. Exiting.")
        sys.exit(2)
    print(f"Universal agent for {COMPETITION_ID}")
    print(f"DATA_PATH={DATA_PATH} OUTPUT_PATH={OUTPUT_PATH}")
    task = infer_task_type(COMPETITION_ID, DATA_PATH)
    print("Inferred task:", task)
    try:
        if task == "image":
            run_image_pipeline(DATA_PATH, COMPETITION_ID, OUTPUT_PATH)
        elif task == "text_classification":
            run_text_pipeline(DATA_PATH, COMPETITION_ID, OUTPUT_PATH)
        elif task == "seq2seq":
            run_seq2seq_pipeline(DATA_PATH, COMPETITION_ID, OUTPUT_PATH)
        elif task == "tabular":
            run_tabular_pipeline(DATA_PATH, COMPETITION_ID, OUTPUT_PATH)
        else:
            print("Unknown task. Creating dummy submission from sample_submission if present.")
            import pandas as pd
            sample = Path(DATA_PATH) / COMPETITION_ID / "sample_submission.csv"
            if sample.exists():
                df = pd.read_csv(sample)
                for col in df.columns:
                    if col != df.columns[0]:
                        df[col] = 0
                df.to_csv(OUTPUT_PATH, index=False)
                print("Dummy submission written to", OUTPUT_PATH)
            else:
                print("No sample_submission found; no output.")
    except Exception:
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
