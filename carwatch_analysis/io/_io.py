import pandas as pd

from carwatch_analysis._types import path_t
from carwatch_analysis.stats import create_unique_night_id


def load_cortisol_samples_reporting_times(path: path_t) -> pd.DataFrame:
    data = pd.read_csv(path)
    data = data.set_index(["subject", "night", "condition", "reporting_type", "sample"])

    td_cols = ["wake_onset", "sample_time", "time_diff_to_wake_onset"]
    date_cols = ["date"]

    data[date_cols] = data[date_cols].apply(pd.to_datetime)
    data[td_cols] = data[td_cols].apply(pd.to_timedelta)

    data = create_unique_night_id(data)
    return data


def load_sensor_awakening_times(path: path_t) -> pd.DataFrame:
    data = pd.read_csv(path)
    # select only the columns we need
    data = data[["subject", "night", "wake_onset"]]
    data = data.set_index(["subject", "night"])
    data = data.add_suffix("_sensor")
    # convert to datetime and normalize time
    data = pd.to_datetime(data["wake_onset_sensor"])
    data -= data.dt.normalize()
    return pd.DataFrame(data)


def convert_cortisol_sample_ids(data: pd.DataFrame) -> pd.DataFrame:
    """Increase all cortisol sample IDs by one to match CAR guidelines (S1 = first sample).

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        Dataframe with cortisol samples.

    """
    index_levels = list(data.index.names)
    data = data.reset_index()
    data["sample"] = "S" + (data["sample"].str.extract(r"S(\d)").astype(int) + 1).astype(str)
    data = data.set_index(index_levels)
    return data
