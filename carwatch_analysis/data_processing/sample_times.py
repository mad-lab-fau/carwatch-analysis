import pandas as pd

__all__ = ["add_naive_sample_times", "sample_times_long_format"]

from scipy.stats import stats


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
        stubnames=["sample_time"],
        i=["subject", "night", "condition", "sample"],
        j="log_type",
        sep="_",
        suffix=r"\w+",
    )

    data = pd.wide_to_long(
        data.reset_index(),
        stubnames=["wake_onset"],
        i=["subject", "night", "condition", "sample", "log_type"],
        j="wake_onset_type",
        sep="_",
        suffix=r"\w+",
    )

    # ensure that all data from all samples are present
    data = data.unstack("sample").dropna().stack()
    # reorder index levels
    data = data.reorder_levels(["subject", "night", "condition", "wake_onset_type", "log_type", "sample"]).sort_index()
    # reorder columns
    data = data[["date", "wake_onset", "sample_time", "cortisol"]]
    return data


def compute_sample_times_parameter(data: pd.DataFrame) -> pd.DataFrame:
    data = data.assign(**{"time_diff_to_wake_onset": data["sample_time"] - data["wake_onset"]})
    data = data.assign(**{"time_diff_min": data["time_diff_to_wake_onset"].dt.total_seconds() / 60})
    return data


def restructure_sample_times_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    index_levels = list(data.index.names)
    index_levels.remove("wake_onset_type")

    data = data.reset_index(["wake_onset_type", "log_type"])
    data = data.assign(**{"log_type": data["wake_onset_type"] + "_" + data["log_type"]})
    data = data.drop(columns="wake_onset_type").reset_index().set_index(index_levels)

    log_type_mapping = {
        "selfreport_naive": "Naive",
        # "selfreport_app": "Mixed",
        "selfreport_selfreport": "Selfreport",
        "app_app": "App",
        "sensor_selfreport": "Sensor_Selfreport",
        "sensor_app": "Sensor_App",
    }

    data = data.rename(log_type_mapping, level="log_type")
    data = data.reindex(list(log_type_mapping.values()), level="log_type")
    return data


def compute_time_diff_to_naive(data: pd.DataFrame) -> pd.DataFrame:
    time_diff_min = data["time_diff_min"].copy()

    # unstack and compute time difference between naive and the other two log types
    time_diff_to_naive = time_diff_min.unstack("log_type").apply(
        lambda df: df - time_diff_min.xs("Naive", level="log_type")
    )

    # stack back and convert to dataframe
    time_diff_to_naive = pd.DataFrame(time_diff_to_naive.stack(), columns=["time_diff_to_naive_min"])
    return data.join(time_diff_to_naive)


def add_delay_group_index(data: pd.DataFrame) -> pd.DataFrame:
    wake_onset_diff = data["time_diff_to_naive_min"].xs("S0", level="sample").round(0)

    bins = [wake_onset_diff.min(), 3, 6, 15, wake_onset_diff.max()]
    wake_onset_diff = wake_onset_diff.unstack("log_type").drop(columns="Naive")
    delay_groups = wake_onset_diff.apply(
        pd.cut, bins=bins, include_lowest=True, labels=["None", "Short", "Moderate", "High"]
    )
    delay_groups.columns = delay_groups.columns.str.lower()
    delay_groups = delay_groups.add_prefix("delay_group_")

    data = data.join(delay_groups)
    delay_cols = list(data.filter(like="delay").columns)
    data = data.reset_index(["sample"]).set_index(delay_cols + ["sample"], append=True)
    return data


def compute_cumulative_sampling_delay(data: pd.DataFrame) -> pd.DataFrame:
    cum_sampling_delay = data["S4"] - data["S0"]
    cum_sampling_delay = pd.DataFrame(cum_sampling_delay, columns=["cum_sampling_delay"])
    return cum_sampling_delay.groupby("log_type").agg(["median", stats.iqr])


def categorize_sampling_adherence(data: pd.DataFrame) -> pd.DataFrame:
    wo_s0_data = data.xs("S0", level="sample")["time_diff_to_naive_min"]
    wo_s0_group = pd.cut(
        wo_s0_data,
        bins=[wo_s0_data.min(), 5, wo_s0_data.max()],
        include_lowest=True,
        labels=["Adherent", "Non-adherent"],
    )
    wo_s0_group.name = "delay_group"
    return pd.DataFrame(wo_s0_data).join(wo_s0_group).set_index("delay_group", append=True)
