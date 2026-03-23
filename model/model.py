import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

try:
    from storage.SupabaseStorage import SupabaseStorage
except ModuleNotFoundError:
    # Supports direct execution: python model.py
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from storage.SupabaseStorage import SupabaseStorage


MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(exist_ok=True)


def get_model_path(factory_id: str) -> Path:
    """Return the pickle file path for a given factory."""
    return MODELS_DIR / f"{factory_id}.pkl"


def save_model(regressor, factory_id: str) -> str:
    """Save a trained model for a specific factory."""
    model_path = get_model_path(factory_id)
    joblib.dump(regressor, model_path)
    return str(model_path)


def load_model(factory_id: str):
    """Load a trained model for a specific factory."""
    model_path = get_model_path(factory_id)
    if not model_path.exists():
        raise FileNotFoundError(f"No trained model for factory {factory_id}. Train first with train_and_evaluate().")
    return joblib.load(model_path)


def predict(factory_id: str, features_df: pd.DataFrame) -> list:
    """Load a trained model and make predictions for a factory."""
    regressor = load_model(factory_id)
    return regressor.predict(features_df).tolist()


def train_and_evaluate(factory_id: str = "factory_123", degree: int = 2) -> None:
    """Train and save a model for a specific factory."""
    storage = SupabaseStorage()
    dataframe = storage.fake_data(factory_id)

    if isinstance(dataframe, str):
        print(f"Warning: {dataframe}. Using fallback fake_data().")
        dataframe = storage.fake_data()

    if isinstance(dataframe, str):
        raise RuntimeError(dataframe)

    if dataframe.empty:
        raise ValueError(f"No data available for factory {factory_id}.")

    x = dataframe.drop(columns=["units_produced"])
    y = dataframe["units_produced"]

    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    regressor = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        LinearRegression(),
    )
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Factory: {factory_id}")
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    comparison = pd.DataFrame(
        {
            "actual": y_test.reset_index(drop=True),
            "predicted": y_pred,
        }
    )
    print("Sample predictions (first 10 rows):")
    print(comparison.head(10))

    model_path = save_model(regressor, factory_id)
    print(f"\nModel saved to: {model_path}")

if __name__ == "__main__":
    train_and_evaluate(factory_id="factory_123", degree=2)




