import pandas as pd

__all__ = ["add_naive_sample_times", "sample_times_long_format"]


def add_naive_sample_times(data: pd.DataFrame) -> pd.DataFrame:
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


def compute_sample_times_parameter(data: pd.DataFrame) -> pd.DataFrame:
    data = data.assign(**{"time_diff_to_wake_onset": data["sample_time"] - data["wake_onset"]})
    data = data.assign(**{"time_rel_min": data["time_diff_to_wake_onset"].dt.total_seconds() / 60})
    data["time_rel_min"] = (
        data.groupby(["subject", "night", "log_type"])["time_rel_min"].diff().fillna(data["time_rel_min"])
    )
    return data
