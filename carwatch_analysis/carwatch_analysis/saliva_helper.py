from typing import Dict, Optional

import pandas as pd
from carwatch_analysis._types import path_t

from biopsykit.stats import StatsPipeline


def import_cortisol_raw(path: path_t) -> pd.DataFrame:
    cort_samples = pd.read_csv(path)
    # assign each night an unique_id to allow repeated measures analyses and insert into dataframe
    cort_samples.insert(2, "night_id", cort_samples["subject"] + "_" + cort_samples["night"].astype(str))
    cort_samples = cort_samples.drop(columns="time")
    cort_samples = cort_samples.set_index(list(cort_samples.columns.drop("cortisol")))
    return cort_samples


def import_cortisol_features(path: path_t) -> pd.DataFrame:
    cort_features = pd.read_csv(path)
    cort_features = cort_features.set_index(list(cort_features.columns[:-1]))
    return cort_features


def analysis_saliva_raw(data: pd.DataFrame, variable: str) -> StatsPipeline:
    pipeline = StatsPipeline(
        steps=[("prep", "normality"), ("prep", "equal_var"), ("test", "mixed_anova"), ("posthoc", "pairwise_ttests")],
        params={
            "dv": "cortisol",
            "between": variable,
            "within": "sample",
            "subject": "night_id",
            "padjust": "bonf",
        },
    )

    pipeline.apply(data)

    return pipeline


def analysis_saliva_features(
    data: pd.DataFrame, variable: str, test_type: Optional[str] = "welch_anova"
) -> StatsPipeline:
    if test_type == "welch_anova":
        posthoc = "pairwise_gameshowell"
    else:
        posthoc = "pairwise_ttests"

    steps = [("prep", "normality"), ("prep", "equal_var"), ("test", test_type), ("posthoc", posthoc)]
    params = {"groupby": "saliva_feature", "dv": "cortisol", "between": variable, "padjust": "bonf"}
    pipeline = StatsPipeline(steps=steps, params=params)

    pipeline.apply(data)
    return pipeline
