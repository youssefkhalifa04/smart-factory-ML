import pandas as pd

def dataframe_generator(data: list[dict]) -> pd.DataFrame:
    """
    Convert a list of dictionaries into a pandas DataFrame.

    Args:
        data (list[dict]): A list where each element is a dictionary representing a row of data.

    Returns:
        pd.DataFrame: A DataFrame constructed from the input data.
    """


    
    return pd.DataFrame(data)


def _season_from_month(month: int) -> str:
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    if month in (9, 10, 11):
        return "autumn"
    return "winter"


def prepare_production_dataframe(data: list[dict], normalize: bool = True) -> pd.DataFrame:
    """
    Build a model-ready dataframe from raw daily production rows.

    Expected input row keys: date, units_produced
    Output columns:
    monday..sunday, spring..winter, lag_3, lag_7, lag_14, units_produced
    """
    df = dataframe_generator(data)

    if df.empty:
        return pd.DataFrame(
            columns=[
                "monday",
                "tuesday",
                "wednesday",
                "thursday",
                "friday",
                "saturday",
                "sunday",
                "spring",
                "summer",
                "autumn",
                "winter",
                "lag_3",
                "lag_7",
                "lag_14",
                "units_produced",
            ]
        )

    required_columns = {"date", "units_produced"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    # Keep only fields required for feature engineering.
    df = df[["date", "units_produced"]].copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["units_produced"] = pd.to_numeric(df["units_produced"], errors="coerce")

    # Remove invalid rows and sort chronologically before lag creation.
    df = df.dropna(subset=["date", "units_produced"]).sort_values("date").reset_index(drop=True)

    weekday_names = [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    ]
    season_names = ["spring", "summer", "autumn", "winter"]

    weekdays = pd.get_dummies(df["date"].dt.day_name().str.lower(), dtype=int)
    weekdays = weekdays.reindex(columns=weekday_names, fill_value=0)

    seasons = pd.get_dummies(df["date"].dt.month.map(_season_from_month), dtype=int)
    seasons = seasons.reindex(columns=season_names, fill_value=0)

    df["lag_3"] = df["units_produced"].shift(3)
    df["lag_7"] = df["units_produced"].shift(7)
    df["lag_14"] = df["units_produced"].shift(14)

    processed = pd.concat(
        [
            weekdays,
            seasons,
            df[["lag_3", "lag_7", "lag_14", "units_produced"]],
        ],
        axis=1,
    )

    # Remove rows containing null values (mostly from lag initialization).
    processed = processed.dropna(axis=0).reset_index(drop=True)

    if normalize and not processed.empty:
        feature_columns = processed.columns[:-1]
        feature_values = processed[feature_columns].astype(float)
        min_values = feature_values.min()
        max_values = feature_values.max()
        denominator = (max_values - min_values).replace(0, 1)
        processed.loc[:, feature_columns] = (feature_values - min_values) / denominator
    

    return processed


def load_model(factory_id: str):
    """Load a trained model for a specific factory."""
    model_path = f"models/{factory_id}.pkl"
    return pd.read_pickle(model_path)

def is_outdated(model_date: pd.Timestamp, threshold_days: int = 7) -> bool:
    """Determine if a model is outdated based on its training date."""
    print(f"Model date: {model_date}")
    print(f"Current date: {pd.Timestamp.now()}")
    if pd.isna(model_date):
        return True
    age = (pd.Timestamp.now() - model_date).days
    return age > threshold_days

def prepare_data(factory_id: str) -> pd.DataFrame:
    """Build one model-ready feature row for next-day prediction."""
    from storage.SupabaseStorage import SupabaseStorage

    storage = SupabaseStorage()
    latest = storage.get_latest_data(factory_id)
    print(f"Latest data for factory {factory_id}: {latest}")

    if isinstance(latest, str) and latest.startswith("DATABASE ERROR"):
        raise RuntimeError(latest)
    if not latest:
        raise ValueError(f"No latest data found for factory {factory_id}.")
    if len(latest) < 14:
        raise ValueError("At least 14 daily records are required to compute lag_14.")

    hist = pd.DataFrame(latest)
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    hist["units_produced"] = pd.to_numeric(hist["units_produced"], errors="coerce")
    hist = hist.dropna(subset=["date", "units_produced"]).sort_values("date").reset_index(drop=True)

    if hist.shape[0] < 14:
        raise ValueError("Not enough valid records after cleaning to compute lags.")

    next_date = hist["date"].iloc[-1] + pd.Timedelta(days=1)
    weekday = next_date.day_name().lower()

    def season_from_month(month: int) -> str:
        if month in (3, 4, 5):
            return "spring"
        if month in (6, 7, 8):
            return "summer"
        if month in (9, 10, 11):
            return "autumn"
        return "winter"

    season = season_from_month(next_date.month)

    # Latest values for lags:
    # lag_3 uses value from t-3, lag_7 from t-7, lag_14 from t-14
    units = hist["units_produced"].tolist()
    lag_3 = float(units[-3])
    lag_7 = float(units[-7])
    lag_14 = float(units[-14])

    # Match your training normalization style approximately (window min-max)
    mn = float(min(units))
    mx = float(max(units))
    denom = (mx - mn) if (mx - mn) != 0 else 1.0

    lag_3 = (lag_3 - mn) / denom
    lag_7 = (lag_7 - mn) / denom
    lag_14 = (lag_14 - mn) / denom

    features_df = pd.DataFrame(
        [{
            "monday": 1.0 if weekday == "monday" else 0.0,
            "tuesday": 1.0 if weekday == "tuesday" else 0.0,
            "wednesday": 1.0 if weekday == "wednesday" else 0.0,
            "thursday": 1.0 if weekday == "thursday" else 0.0,
            "friday": 1.0 if weekday == "friday" else 0.0,
            "saturday": 1.0 if weekday == "saturday" else 0.0,
            "sunday": 1.0 if weekday == "sunday" else 0.0,
            "spring": 1.0 if season == "spring" else 0.0,
            "summer": 1.0 if season == "summer" else 0.0,
            "autumn": 1.0 if season == "autumn" else 0.0,
            "winter": 1.0 if season == "winter" else 0.0,
            "lag_3": lag_3,
            "lag_7": lag_7,
            "lag_14": lag_14,
        }]
    )

    return features_df