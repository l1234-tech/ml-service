import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402

from ml.config import PIPELINE_PATH, SAMPLE_PAYLOAD_PATH  # noqa: E402
from ml.utils import load_object  # noqa: E402


def test_load_pipeline():
    pipeline = load_object(PIPELINE_PATH)
    assert pipeline is not None
    print("Pipeline loaded successfully!")


def test_predict_sample():
    pipeline = load_object(PIPELINE_PATH)
    sample = pd.read_json(SAMPLE_PAYLOAD_PATH)
    preds = pipeline.predict(sample)
    proba = pipeline.predict_proba(sample)
    assert len(preds) == len(sample)
    assert proba.shape[0] == len(sample)
    print(f"Predictions: {preds}")
    print(f"Probabilities: {proba[:, 1]}")


if __name__ == "__main__":
    test_load_pipeline()
    test_predict_sample()
