import os
from utils.dataset_loader import detect_dataset_type
from pipelines.image_pipeline import train_image_model
from pipelines.text_pipeline import train_text_model
from pipelines.tabular_pipeline import train_tabular_model

DATA_PATH = os.getenv("DATA_PATH")

def main():
    dtype = detect_dataset_type(DATA_PATH)
    print("Detected Dataset Type:", dtype)

    if dtype == "image":
        train_image_model(DATA_PATH)
    elif dtype == "text":
        train_text_model(DATA_PATH)
    elif dtype == "tabular":
        train_tabular_model(DATA_PATH)
    else:
        raise ValueError("Unknown dataset type detected")

if __name__ == "__main__":
    main()
