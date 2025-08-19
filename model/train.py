import json
import os
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(BASE_DIR, "model", "artifacts")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "model.joblib")
SCHEMA_PATH = os.path.join(ARTIFACT_DIR, "schema.json")


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
	"""Create a ColumnTransformer with separate encoder pipelines per categorical feature."""
	# Numeric pipeline
	numeric_pipeline = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="median")),
		]
	)

	# Separate pipelines per categorical feature
	categorical_pipelines: List[Tuple[str, Pipeline, List[str]]] = []
	for feature_name in categorical_features:
		pipe = Pipeline(
			steps=[
				("imputer", SimpleImputer(strategy="most_frequent")),
				("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
			]
		)
		categorical_pipelines.append((f"{feature_name}_cat", pipe, [feature_name]))

	# Combine into a single ColumnTransformer
	transformers: List[Tuple[str, Any, Any]] = [("numeric", numeric_pipeline, numeric_features)] + categorical_pipelines

	preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
	return preprocessor


def load_data() -> pd.DataFrame:
	"""Load Titanic dataset using seaborn."""
	df = sns.load_dataset("titanic")
	return df


def train_and_save(random_state: int = 42) -> Tuple[Any, Dict[str, Any]]:
	os.makedirs(ARTIFACT_DIR, exist_ok=True)

	df = load_data()

	# Define features and target
	predictors: List[str] = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
	target = "survived"

	# Filter only required columns
	df = df[predictors + [target]].copy()

	# Drop rows with missing target
	df = df.dropna(subset=[target])

	# Train/test split for a quick quality check
	X_train, X_test, y_train, y_test = train_test_split(
		df[predictors], df[target], test_size=0.2, random_state=random_state, stratify=df[target]
	)

	numeric_features = ["age", "sibsp", "parch", "fare", "pclass"]
	categorical_features = ["sex", "embarked"]

	preprocessor = build_preprocessor(numeric_features, categorical_features)

	# Model
	rf = RandomForestClassifier(
		n_estimators=400,
		random_state=random_state,
		max_depth=None,
		min_samples_split=2,
		min_samples_leaf=1,
		n_jobs=-1,
	)

	pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", rf)])

	# Fit
	pipeline.fit(X_train, y_train)

	# Quick evaluation
	y_pred = pipeline.predict(X_test)
	acc = float(accuracy_score(y_test, y_pred))
	print(f"Validation accuracy: {acc:.3f}")

	# Persist artifacts
	joblib.dump(pipeline, MODEL_PATH)

	# Build schema and persist
	schema: Dict[str, Any] = {
		"target": target,
		"predictors": predictors,
		"numeric_features": numeric_features,
		"categorical_features": categorical_features,
		"categories": {
			"sex": sorted([x for x in df["sex"].dropna().unique().tolist() if isinstance(x, str)]),
			"embarked": sorted([x for x in df["embarked"].dropna().unique().tolist() if isinstance(x, str)]),
		},
		"metrics": {"accuracy": acc},
	}

	with open(SCHEMA_PATH, "w", encoding="utf-8") as f:
		json.dump(schema, f, indent=2)

	return pipeline, schema


if __name__ == "__main__":
	train_and_save()
