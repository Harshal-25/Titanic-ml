import os
import json
from typing import Any, Dict, Tuple

import joblib
import pandas as pd
from flask import Flask, render_template, request


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "model", "artifacts")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "model.joblib")
SCHEMA_PATH = os.path.join(ARTIFACT_DIR, "schema.json")


def load_or_train() -> Tuple[Any, Dict[str, Any]]:
	"""Load trained pipeline and schema; train if missing."""
	model = None
	schema: Dict[str, Any] = {}
	try:
		model = joblib.load(MODEL_PATH)
		with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
			schema = json.load(f)
	except FileNotFoundError:
		# Train on the fly if artifacts are missing
		from model.train import train_and_save

		model, schema = train_and_save()
	except Exception:
		# If artifact loading fails for any other reason, try retraining
		from model.train import train_and_save

		model, schema = train_and_save()

	return model, schema


app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "dev")
model_pipeline, feature_schema = load_or_train()


@app.route("/", methods=["GET"])  # type: ignore[misc]
def index():
	return render_template(
		"index.html",
		schema=feature_schema,
		result=None,
	)


@app.route("/predict", methods=["POST"])  # type: ignore[misc]
def predict():
	# Extract inputs from form, coerce types, let imputers handle missing
	form = request.form

	def to_float(value: str):
		try:
			return float(value)
		except Exception:
			return None

	def to_int(value: str):
		try:
			return int(float(value))
		except Exception:
			return None

	payload = {
		"pclass": to_int(form.get("pclass", "")),
		"sex": (form.get("sex", "") or None),
		"age": to_float(form.get("age", "")),
		"sibsp": to_int(form.get("sibsp", "")),
		"parch": to_int(form.get("parch", "")),
		"fare": to_float(form.get("fare", "")),
		"embarked": (form.get("embarked", "") or None),
	}

	# Create DataFrame with expected columns
	X = pd.DataFrame([payload], columns=feature_schema["predictors"])  # type: ignore[index]

	proba = model_pipeline.predict_proba(X)[0][1]
	prediction = int(proba >= 0.5)

	result = {
		"probability": round(float(proba) * 100.0, 2),
		"prediction": prediction,
	}

	return render_template(
		"index.html",
		schema=feature_schema,
		result=result,
		inputs=payload,
	)


if __name__ == "__main__":
	app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
