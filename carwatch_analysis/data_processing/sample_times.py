from typing import Sequence

import pandas as pd

__all__ = [
    "add_naive_sample_times",
    "sample_times_long_format",
    "compute_sample_times_parameter",
    "restructure_sample_times_dataframe",
    "compute_cumulative_sampling_delay",
    "compute_sample_times_parameter",
    "categorize_sampling_adherence",
    "compute_max_delay_selfreport",
    "compute_time_diff_to_naive",
]

from scipy.stats import stats


def add_naive_sample_times(data: pd.DataFrame) -> pd.DataFrame:
    """Add naive sampling times to the data.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        Dataframe containing cortisol values and sampling times.


    Returns
    -------
    :class:`pandas.DataFrame`
        Dataframe containing cortisol values and sampling times with naive sampling times added.

    """
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
    """Convert dataframe with sampling times into long-format.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        Dataframe containing sampling times.


    Returns
    -------
    :class:`pandas.DataFrame`
        Dataframe containing sampling times in long-format.

    """
    data = pd.wide_to_long(
        data.reset_index(),
        stubnames=["sample_time"],
        i=["subject", "night", "condition", "sample"],
        j="reporting_type",
        sep="_",
        suffix=r"\w+",
    )

    data = pd.wide_to_long(
        data.reset_index(),
        stubnames=["wake_onset"],
        i=["subject", "night", "condition", "sample", "reporting_type"],
        j="wake_onset_type",
        sep="_",
        suffix=r"\w+",
    )

    # ensure that all data from all samples are present
    data = data.unstack("sample").dropna().stack()
    # reorder index levels
    data = data.reorder_levels(
        ["subject", "night", "condition", "wake_onset_type", "reporting_type", "sample"]
    ).sort_index()
    # reorder columns
    data = data[["date", "wake_onset", "sample_time", "cortisol"]]
    return data


def compute_sample_times_parameter(data: pd.DataFrame) -> pd.DataFrame:
    """Compute sample-wise difference to wake onset (in min) for each reporting strategy.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        Dataframe containing sampling times.


    Returns
    -------
    :class:`pandas.DataFrame`
        Dataframe containing sample-wise difference to wake onset (in min) for each reporting strategy.

    """
    data = data.assign(**{"time_diff_to_wake_onset": data["sample_time"] - data["wake_onset"]})
    data = data.assign(**{"time_diff_min": data["time_diff_to_wake_onset"].dt.total_seconds() / 60})
    return data


def restructure_sample_times_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """Restructure dataframe containing sampling times.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        Dataframe containing sampling times.


    Returns
    -------
    :class:`pandas.DataFrame`
        Restructured dataframe containing sampling times.

    """
    index_levels = list(data.index.names)
    index_levels.remove("wake_onset_type")

    data = data.reset_index(["wake_onset_type", "reporting_type"])
    data = data.assign(**{"reporting_type": data["wake_onset_type"] + "_" + data["reporting_type"]})
    data = data.drop(columns="wake_onset_type").reset_index().set_index(index_levels)

    reporting_type_mapping = {
        "selfreport_naive": "Naive",
        "selfreport_selfreport": "AW & ST: Selfreport",
        "app_app": "AW & ST: App",
        "sensor_naive": "AW: Sensor, ST: Naive",
        "sensor_selfreport": "AW: Sensor, ST: Selfreport",
        "sensor_app": "AW: Sensor, ST: App",
    }

    data = data.rename(reporting_type_mapping, level="reporting_type")
    data = data.reindex(list(reporting_type_mapping.values()), level="reporting_type")
    return data


def compute_time_diff_to_naive(data: pd.DataFrame) -> pd.DataFrame:
    """Compute time difference to naive sampling times.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        Dataframe containing sampling times.


    Returns
    -------
    :class:`pandas.DataFrame`
        Dataframe containing time difference to naive sampling times.
    """
    time_diff_min = data["time_diff_min"].copy()

    # unstack and compute time difference between naive and the other two reporting types
    time_diff_to_naive = time_diff_min.unstack("reporting_type").apply(
        lambda df: df - time_diff_min.xs("Naive", level="reporting_type")
    )

    # stack back and convert to dataframe
    time_diff_to_naive = pd.DataFrame(time_diff_to_naive.stack(), columns=["time_diff_to_naive_min"])
    return data.join(time_diff_to_naive)


def add_delay_group_index(data: pd.DataFrame) -> pd.DataFrame:
    """Compute the "delay group" for each recording and add it as index to the dataframe.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        Dataframe containing sampling times.


    Returns
    -------
    :class:`pandas.DataFrame`
        Dataframe containing sampling times with "delay group" index.

    """
    wake_onset_diff = data["time_diff_to_naive_min"].xs("S1", level="sample").round(0)

    bins = [wake_onset_diff.min(), 3, 6, 15, wake_onset_diff.max()]
    wake_onset_diff = wake_onset_diff.unstack("reporting_type").drop(columns="Naive")
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
    """Compute cumulative sampling delay, i.e, delay between the first (S1) and last (S5) saliva sample.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        Dataframe containing sampling times.

    Returns
    -------
    :class:`pandas.DataFrame`
        Dataframe containing cumulative sampling delay.

    """
    cum_sampling_delay = data["S5"] - data["S1"]
    cum_sampling_delay = pd.DataFrame(cum_sampling_delay, columns=["cum_sampling_delay"])
    return cum_sampling_delay.groupby("reporting_type").agg(["median", stats.iqr])


def categorize_sampling_adherence(data: pd.DataFrame) -> pd.DataFrame:
    """Classify sampling adherence for each recording.

    Recordings are classified as adherent if the delay between awakening and S1 is less than 5 min.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        Dataframe containing sampling times.

    Returns
    -------
    :class:`pandas.DataFrame`
        Dataframe containing sampling adherence.

    """
    wo_s1_data = data.xs("S1", level="sample")["time_diff_to_naive_min"]
    wo_s1_group = pd.cut(
        wo_s1_data,
        bins=[wo_s1_data.min(), 5, wo_s1_data.max()],
        include_lowest=True,
        labels=["Adherent", "Non-adherent"],
    )
    wo_s1_group.name = "delay_group"
    return pd.DataFrame(wo_s1_data).join(wo_s1_group).set_index("delay_group", append=True)


def compute_max_delay_selfreport(data: pd.DataFrame, ids_max_delay: Sequence[str]) -> pd.DataFrame:
    max_delay_selfreport = data["time_diff_to_naive_min"].reindex(ids_max_delay, level="night_id")
    max_delay_selfreport = max_delay_selfreport.reindex(
        ["AW & ST: Selfreport", "AW & ST: App"], level="reporting_type"
    ).unstack(["sample", "reporting_type"])
    max_delay_selfreport = (
        max_delay_selfreport.sort_index(axis=1).dropna().sort_values(by=("S1", "AW & ST: App"), ascending=False)
    )
    max_delay_selfreport = max_delay_selfreport.round(2)
    return max_delay_selfreport
