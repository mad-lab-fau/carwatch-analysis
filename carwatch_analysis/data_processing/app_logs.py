import json
from typing import Optional

import pandas as pd
from biopsykit.carwatch_logs import LogData, log_actions
from biopsykit.carwatch_logs.log_data import get_logs_for_action

from carwatch_analysis.datasets import CarWatchDatasetProcessed

__all__ = ["process_app_log_single_subject"]


def process_app_log_single_subject(dataset: CarWatchDatasetProcessed) -> Optional[pd.DataFrame]:
    app_logs = dataset.app_logs
    log_data = LogData(app_logs)
    # add 1 day because it's the next morning
    recording_days = dataset.date + pd.Timedelta("1 day")
    recording_days["date"] = recording_days["date"].dt.tz_localize("Europe/Berlin")
    finished_days = [day.normalize() for day in log_data.finished_days]

    df_barcode = get_logs_for_action(log_data, log_actions.barcode_scanned)

    dict_nights = {}
    for night_id, day in enumerate(recording_days["date"]):
        if day not in finished_days:
            continue

        day_mask = df_barcode.index.normalize().isin([day])
        df_barcode_night = df_barcode.loc[day_mask]

        df_barcode_night = df_barcode_night.assign(
            **{"sample": df_barcode_night["extras"].apply(_get_saliva_id_from_json)}
        )
        df_barcode_night = df_barcode_night.set_index("sample", append=True)
        df_barcode_night = df_barcode_night.drop("S5", level="sample", errors="ignore")

        if df_barcode_night.empty:
            continue

        dict_nights[night_id] = df_barcode_night

    if len(dict_nights) == 0:
        return None
    return pd.concat(dict_nights, names=["night"])


def _get_saliva_id_from_json(col: str) -> str:
    json_extra = json.loads(col)
    return f"S{json_extra.get('saliva_id')}"


def restructure_sampling_times_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    data = data.drop(columns=["action", "extras"])
    data = data.reset_index("time")
    data = data.rename(columns={"time": "sample_time_app"})
    data["sample_time_app"] -= data["sample_time_app"].dt.normalize()
    data = data.loc[~data.index.duplicated(keep="last")]
    return data


def add_naive_times(data: pd.DataFrame) -> pd.DataFrame:
    data = data.assign(**{"wake_onset_naive": data["wake_onset_selfreport"]})

    sample_ids = data.index.get_level_values("sample").unique()
    sample_time_naive = pd.Series(
        [i * 15 for i in range(0, len(sample_ids))], index=sample_ids, name="sample_time_naive"
    )
    sample_time_naive = pd.to_timedelta(sample_time_naive, unit="min")

    data = data.join(sample_time_naive)
    data["sample_time_naive"] += data["wake_onset_naive"]
    return data


def sample_times_long_format(data: pd.DataFrame) -> pd.DataFrame:
    data = pd.wide_to_long(
        data.reset_index(),
        stubnames=["sample_time", "wake_onset"],
        i=["subject", "night", "condition", "sample"],
        j="log_type",
        sep="_",
        suffix=r"\w+",
    )

    # ensure that all data from all samples are present
    data = data.unstack("sample").dropna().stack()

    # reorder index levels
    data = data.reorder_levels(["subject", "night", "condition", "log_type", "sample"]).sort_index()
    # reorder columns
    data = data[["date", "wake_onset", "sample_time", "cortisol"]]
    return data
