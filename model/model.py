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

try: 
    from utils.utils import is_outdated 
except ModuleNotFoundError:
    # Supports direct execution: python model.py
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from utils.utils import is_outdated 
storage = SupabaseStorage()
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
    p = regressor.predict(features_df)
    print(f"Predicted production for next day: {p[0]:.2f} units")
    return p.tolist()


def train_and_evaluate(factory_id: str = "97e90fd2-469a-471b-a824-1e6ac0d5ec93", degree: int = 2, features_df: pd.DataFrame = None) -> None:
    """Train and save a model for a specific factory."""
    global storage
    dataframe = storage.get_data(factory_id)

    if isinstance(dataframe, str):
        print(f"Warning: {dataframe}. Using fallback fake_data().")
        dataframe = storage.get_data(factory_id)

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

    
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    p = regressor.predict(features_df)
    print(f"Predicted production for next day: {p[0]:.2f} units")
    model_path = save_model(regressor, factory_id)
    print(f"\nModel saved to: {model_path}")

def run(factory_id: str, degree: int = 2 , features_df: pd.DataFrame = None):
    """Predict if the model isn't outdated, otherwise retrain and predict."""
    global storage
    try:
        model_date_str = storage.last_trained_model(factory_id)
        print(f"Last trained model date string for factory {factory_id}: {model_date_str}")
        model_date = pd.to_datetime(model_date_str, errors="coerce")
        if is_outdated(model_date):
            print(f"Model for factory {factory_id} is outdated. Retraining...")
            train_and_evaluate(factory_id, degree, features_df)
        else:
            print(f"Model for factory {factory_id} is up-to-date. Making predictions...")
            predict(factory_id, features_df)
    except Exception as e:
        print(f"Error checking model date: {e}. Proceeding to retrain.")
        train_and_evaluate(factory_id, degree, features_df)


    
    
if __name__ == "__main__":
    features_df = pd.DataFrame([
        {
            "monday": 1.0,
            "tuesday": 0.0,
            "wednesday": 0.0,
            "thursday": 0.0,
            "friday": 0.0,
            "saturday": 0.0,
            "sunday": 0.0,
            "spring": 0.0,
            "summer": 1.0,
            "autumn": 0.0,
            "winter": 0.0,
            "lag_3": 0.55,
            "lag_7": 0.61,
            "lag_14": 0.49,
        }
    ])
    run(factory_id="97e90fd2-469a-471b-a824-1e6ac0d5ec93", degree=2 , features_df=features_df)




