import argparse
from src.data_utils import create_demo_dataset
from src.model import train_and_save
from config import DATASET_PATH


def main():
    parser = argparse.ArgumentParser(description="Train the AI-Powered Smart Flood Warning System Random Forest model.")
    parser.add_argument("--make-demo-data", action="store_true", help="Create synthetic sample data before training.")
    args = parser.parse_args()
    if args.make_demo_data or not DATASET_PATH.exists():
        print("Creating synthetic demonstration data. Replace it with a verified dataset for meaningful research results.")
        create_demo_dataset()
    metrics = train_and_save()
    print(f"Model saved. Hold-out accuracy on this dataset: {metrics['accuracy']:.1%}")
    print("Reminder: performance on synthetic data is not evidence of real-world flood prediction accuracy.")


if __name__ == "__main__":
    main()
