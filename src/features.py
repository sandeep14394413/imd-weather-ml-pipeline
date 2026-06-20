import pandas as pd


def add_calendar_features(frame: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    data = frame.copy()
    dates = pd.to_datetime(data[date_column])
    data["day_of_year"] = dates.dt.dayofyear
    data["month"] = dates.dt.month
    data["year"] = dates.dt.year
    data["week"] = dates.dt.isocalendar().week.astype(int)
    return data


def build_training_frame(raw_daily_state_data: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "state", "tmax_c"}
    missing = required - set(raw_daily_state_data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return add_calendar_features(raw_daily_state_data)
