from typing import Sequence

import pandas as pd


def compute_time_unit_digits(data: pd.DataFrame) -> pd.DataFrame:
    """Compute the time unit digits for given awakening and sampling times.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        Dataframe with awakening and sampling times.

    Returns
    -------
    :class:`pandas.DataFrame`
        Dataframe with time unit digits.

    """
    data = data.assign(
        **{
            "sample_minute": data["sample_time"].dt.components["minutes"] % 10,
            "wake_onset_minute": data["wake_onset"].dt.components["minutes"] % 10,
        }
    )
    return data


def time_unit_digit_statistics(data: pd.DataFrame, reporting_type_order: Sequence[str]) -> pd.DataFrame:
    """Compute statistics for time unit digits.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        Dataframe with awakening and sampling times.
    reporting_type_order : list of str
        Order of reporting types to appear in the output.

    Returns
    -------
    :class:`pandas.DataFrame`
        Dataframe with statistics for time unit digits.

    """
    data = data[["wake_onset_minute", "sample_minute"]]
    data = data.reindex(reporting_type_order, level="reporting_type").stack()
    data.index = data.index.set_names("type", level=-1)
    data = pd.DataFrame(data, columns=["time"])
    data = data.groupby(["condition", "reporting_type", "type"]).value_counts(normalize=True) * 100
    data = data.unstack("time").fillna(0).round(0)

    return data
