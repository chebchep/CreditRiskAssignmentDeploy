
import json
from pathlib import Path

import numpy as np
import pandas as pd
import onnx
import onnxruntime as ort

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

SEED = 20
BASE_DIR = Path(__file__).resolve().parent

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

df = pd.read_excel(BASE_DIR / "cc.xls", header=1)

X = df.drop(columns=["ID", "default payment next month"])
y = df["default payment next month"]

# --------------------------------------------------
# Train/Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)

# --------------------------------------------------
# Tuned Random Forest from Module 8
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    min_samples_split=10,
    max_depth=20,
    class_weight="balanced",
    random_state=SEED,
    n_jobs=1
)

print("Training Random Forest...")

model.fit(X_train, y_train)

# --------------------------------------------------
# Evaluate Model
# --------------------------------------------------

y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --------------------------------------------------
# Convert Random Forest to ONNX
# --------------------------------------------------

initial_type = [
    ("float_input", FloatTensorType([None, X.shape[1]]))
]

onnx_model = convert_sklearn(
    model,
    initial_types=initial_type,
    options={id(model): {"zipmap": False}},
    target_opset=17
)

onnx.checker.check_model(onnx_model)

model_path = BASE_DIR / "model.onnx"

with open(model_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print("\nONNX model exported successfully.")

# --------------------------------------------------
# Save Feature Order
# --------------------------------------------------

metadata = {
    "features": X.columns.tolist(),
    "classes": model.classes_.tolist(),
    "hyperparameters": model.get_params()
}

with open(BASE_DIR / "model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

# --------------------------------------------------
# Verify ONNX Model
# --------------------------------------------------

session = ort.InferenceSession(
    str(model_path),
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

onnx_results = session.run(
    None,
    {input_name: X_test.to_numpy(dtype=np.float32)}
)

onnx_predictions = onnx_results[0].flatten()
onnx_probabilities = onnx_results[1]

original_probabilities = model.predict_proba(X_test)

matching_accuracy = np.mean(y_pred == onnx_predictions)

print("\nONNX Verification:")
print(f"Prediction agreement: {matching_accuracy:.4%}")

print(
    "Maximum probability difference:",
    np.max(np.abs(original_probabilities - onnx_probabilities))
)

if np.array_equal(y_pred, onnx_predictions) and np.allclose(
    original_probabilities,
    onnx_probabilities,
    atol=1e-5
):
    print("PASSED: ONNX model matches the original model.")
else:
    print("WARNING: Prediction differences detected.")

print("\nModel size:", round(model_path.stat().st_size / 1024**2, 2), "MB")
print("Export process completed.")
