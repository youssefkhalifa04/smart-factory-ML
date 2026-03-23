from interfaces.Storage import Storage
from integration.supabase_client import sp
from utils.utils import prepare_production_dataframe
from pathlib import Path

import pandas as pd

class SupabaseStorage(Storage):

    def __init__(self):
        super().__init__()

    def get_data(self, factory_id: str):
        try:
            data = (
                sp.table("daily_production")
                .select("factory_id, date, units_produced")
                .eq("factory_id", factory_id)
                .execute()
            )

            rows = data.data or []
            return prepare_production_dataframe(rows, normalize=True)
        except Exception as e:
            return f"DATABASE ERROR: {e}"
    
    def push_notif(self, factory_id: str , notification: dict) -> bool:
        try:
            sp.table("notifications").insert({
                "factory_id": factory_id,
                "type": notification["type"],
                "statement": notification["statement"],
            }).execute()
            return True
        except Exception as e:
            print(f"DATABASE ERROR: {e}")
            return False
    def fake_data(self, factory_id: str):
        try:
            csv_path = Path(__file__).resolve().parents[1] / "model" / "dataset_example.csv"
            dataframe = pd.read_csv(csv_path)

            # Raw daily production format.
            if {"date", "units_produced"}.issubset(dataframe.columns):
                rows = dataframe.to_dict(orient="records")
                return prepare_production_dataframe(rows, normalize=True)

            # Already engineered format (weekday/season/lags + target).
            if "units_produced" not in dataframe.columns:
                raise ValueError("CSV must contain units_produced column.")

            dataframe = dataframe.dropna(axis=0).reset_index(drop=True)
            if dataframe.empty:
                return dataframe

            feature_columns = dataframe.columns[:-1]
            feature_values = dataframe[feature_columns]
            min_values = feature_values.min()
            max_values = feature_values.max()
            denominator = (max_values - min_values).replace(0, 1)
            dataframe.loc[:, feature_columns] = (feature_values - min_values) / denominator

            return dataframe
        except Exception as e:
            return f"DATA PREPARATION ERROR: {e}"

 