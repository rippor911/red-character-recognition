from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAIN_LABELS = ROOT / "dataset" / "train" / "labels.csv"
SUBMISSION_SAMPLE = ROOT / "dataset" / "submission_sample.csv"
OUTPUT_DIR = ROOT / "outputs"
SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"


def load_train_data():
    return pd.read_csv(TRAIN_LABELS)


def load_test_data():
    return pd.read_csv(SUBMISSION_SAMPLE)


def train_model(train_df):
    return None


def predict_test(model, test_df):
    return ["00000"] * len(test_df)


def save_submission(test_df, predictions):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame({"id": test_df["id"], "label": predictions})
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved submission to {SUBMISSION_PATH}")


def main():
    train_df = load_train_data()
    model = train_model(train_df)
    test_df = load_test_data()
    predictions = predict_test(model, test_df)
    save_submission(test_df, predictions)


if __name__ == "__main__":
    main()
