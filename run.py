import os
from utils.dataset_loader import detect_competition, load_dataset
from utils.predictor import save_predictions
from pipelines.image_pipeline import run_image_pipeline
from pipelines.text_pipeline import run_text_pipeline
from pipelines.tabular_pipeline import run_tabular_pipeline

DATA_PATH = os.getenv("DATA_PATH")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "submission.csv")

def main():
    comp_id = detect_competition(DATA_PATH)
    print(f"Detected competition: {comp_id}")

    train_df, test_df, extra = load_dataset(comp_id, DATA_PATH)

    if "image" in comp_id:
        preds = run_image_pipeline(train_df, test_df, extra)
    elif "text" in comp_id:
        preds = run_text_pipeline(train_df, test_df, extra)
    else:
        preds = run_tabular_pipeline(train_df, test_df, extra)

    save_predictions(preds, OUTPUT_PATH)
    print("Submission saved at:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
