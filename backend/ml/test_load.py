import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402
from ml.config import PIPELINE_PATH, SAMPLE_PAYLOAD_PATH  # noqa: E402
from ml.utils import load_object, load_json  # noqa: E402


def main():
    print("=" * 60)
    print("Pipeline Load Test")
    print("=" * 60)

    try:
        print(f"\n1. Loading pipeline from {PIPELINE_PATH}")
        pipeline = load_object(PIPELINE_PATH)
        print("   Pipeline loaded successfully!")

        print(f"\n2. Loading sample payload from {SAMPLE_PAYLOAD_PATH}")
        sample_data = load_json(SAMPLE_PAYLOAD_PATH)
        df_sample = pd.DataFrame(sample_data)
        print(f"   Sample shape: {df_sample.shape}")
        print("   Sample data:")
        print(df_sample)

        print("\n3. Making predictions")
        predictions = pipeline.predict(df_sample)
        probabilities = pipeline.predict_proba(df_sample)

        print(f"   Predictions: {predictions}")
        print(f"   Probabilities (class 1): {probabilities[:, 1]}")

        print("\n" + "=" * 60)
        print("Test passed successfully!")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"\n   ERROR: File not found - {e}")
        print("   Make sure you have run train.py first!")
        sys.exit(1)
    except Exception as e:
        print(f"\n   ERROR: {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()