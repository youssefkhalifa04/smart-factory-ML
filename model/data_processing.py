from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


def preprocess_dataset(csv_path: str | Path) -> Tuple[pd.DataFrame, pd.Series]:
	"""
	Load and preprocess a dataset for supervised learning.

	Rules applied:
	- Drop rows that contain null values.
	- Normalize feature columns to [0, 1].
	- Use the last column as the prediction target (not normalized).

	Returns:
		X_normalized: normalized feature dataframe
		y: target series (last column)
	"""
	dataset = pd.read_csv(csv_path)

	# Remove any row that contains at least one missing value.
	dataset = dataset.dropna(axis=0).reset_index(drop=True)

	if dataset.shape[1] < 2:
		raise ValueError("Dataset must contain at least one feature column and one target column.")

	feature_columns = dataset.columns[:-1]
	target_column = dataset.columns[-1]

	X = dataset[feature_columns].copy()
	y = dataset[target_column].copy()

	# Min-max normalization with protection against constant-value columns.
	min_values = X.min()
	max_values = X.max()
	denominator = (max_values - min_values).replace(0, 1)
	X_normalized = (X - min_values) / denominator

	return X_normalized, y


if __name__ == "__main__":
	current_dir = Path(__file__).resolve().parent
	input_csv = current_dir / "dataset_example.csv"

	X_processed, y_target = preprocess_dataset(input_csv)
	print("Features shape:", X_processed.shape)
	print("Target shape:", y_target.shape)
	print("First 5 processed feature rows:")
	print(X_processed.head())
