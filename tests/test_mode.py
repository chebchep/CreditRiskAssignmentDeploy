
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

BASE_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def session():
    return ort.InferenceSession(
        str(BASE_DIR / "model.onnx"),
        providers=["CPUExecutionProvider"]
    )


def test_model_loading(session):
    assert len(session.get_inputs()) == 1
    assert len(session.get_outputs()) == 2


def test_input_dimensions(session):
    assert session.get_inputs()[0].shape[1] == 23


def test_metadata():
    with open(BASE_DIR / "model_metadata.json") as f:
        metadata = json.load(f)

    assert len(metadata["features"]) == 23
    assert metadata["classes"] == [0, 1]


@pytest.mark.parametrize(
    "sample, expected_class, expected_score",
    [
        (
            [
                200000, 1, 2, 2, 30,
                0, 0, 0, 0, 0, 0,
                10000, 11000, 12000, 13000, 14000, 15000,
                5000, 5000, 5000, 5000, 5000, 5000
            ],
            0, 0.1235
        ),
        (
            [
                50000, 2, 2, 1, 35,
                2, 2, 2, 2, 2, 2,
                45000, 44000, 43000, 42000, 41000, 40000,
                0, 0, 0, 0, 0, 0
            ],
            1, 0.8832
        )
    ]
)
def test_prediction(session, sample, expected_class, expected_score):
    input_name = session.get_inputs()[0].name
    input_data = np.array([sample], dtype=np.float32)

    labels, probabilities = session.run(
        None,
        {input_name: input_data}
    )

    # Check prediction
    assert int(labels[0]) == expected_class

    # Check probability validity
    assert probabilities.shape == (1, 2)
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)
    assert np.isclose(probabilities.sum(), 1.0)

    # Check prediction consistency
    assert np.isclose(
        probabilities[0][1],
        expected_score,
        atol=0.0001
    )
