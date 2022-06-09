from typing import Optional

import pandas as pd
from biopsykit.stats import StatsPipeline
from biopsykit.utils._datatype_validation_helper import _assert_has_index_levels


def create_unique_night_id(data: pd.DataFrame) -> pd.DataFrame:
    _assert_has_index_levels(data, ["subject", "night"], match_atleast=True)
    # alternative cort_features.index.map("{0[0]}_{0[1]}".format
    index_names = list(data.index.names)
    data = data.reset_index()
    data.insert(2, "night_id", data["subject"] + "_" + data["night"].astype(str))
    index_names.insert(2, "night_id")
    data = data.set_index(index_names)
    return data


def stats_pipeline_saliva_samples(data: pd.DataFrame, variable: str) -> StatsPipeline:
    steps = [("prep", "normality"), ("prep", "equal_var"), ("test", "mixed_anova"), ("posthoc", "pairwise_ttests")]
    params = {
        "dv": "cortisol",
        "between": variable,
        "within": "sample",
        "subject": "night_id",
        "multicomp": {"method": "bonf", "levels": True},
    }
    pipeline = StatsPipeline(steps=steps, params=params)

    pipeline.apply(data)
    return pipeline


def stats_pipeline_saliva_features(
    data: pd.DataFrame, variable: str, equal_var: Optional[bool] = True
) -> StatsPipeline:
    if equal_var:
        test_type = "anova"
        posthoc = "pairwise_ttests"
    else:
        test_type = "welch_anova"
        posthoc = "pairwise_gameshowell"

    steps = [("prep", "normality"), ("prep", "equal_var"), ("test", test_type), ("posthoc", posthoc)]
    params = {
        "dv": "cortisol",
        "between": variable,
        "groupby": "saliva_feature",
        "multicomp": {"method": "bonf"},
    }
    pipeline = StatsPipeline(steps=steps, params=params)

    pipeline.apply(data)
    return pipeline


def stats_pipeline_imu_features(data: pd.DataFrame, variable: str) -> StatsPipeline:

    pipeline = StatsPipeline(
        steps=[
            ("prep", "normality"),
            ("prep", "equal_var"),
            ("test", "welch_anova"),
            ("posthoc", "pairwise_gameshowell"),
        ],
        params={
            "groupby": "imu_feature",
            "dv": "data",
            "between": variable,
        },
    )

    pipeline.apply(data)
    return pipeline
