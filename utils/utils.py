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

    weekdays = pd.get_dummies(df["date"].dt.day_name().str.lower())
    weekdays = weekdays.reindex(columns=weekday_names, fill_value=0)

    seasons = pd.get_dummies(df["date"].dt.month.map(_season_from_month))
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
        feature_values = processed[feature_columns]
        min_values = feature_values.min()
        max_values = feature_values.max()
        denominator = (max_values - min_values).replace(0, 1)
        processed.loc[:, feature_columns] = (feature_values - min_values) / denominator

    return processed