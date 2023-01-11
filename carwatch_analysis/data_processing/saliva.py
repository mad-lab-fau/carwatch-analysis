import biopsykit as bp
import numpy as np
import pandas as pd

__all__ = ["compute_saliva_features"]


def compute_saliva_features(data: pd.DataFrame) -> pd.DataFrame:
    cort_feat = [
        bp.saliva.auc(data, saliva_type="cortisol"),
        bp.saliva.max_increase(data),
        bp.saliva.slope(data, sample_idx=[0, 3]),
        bp.saliva.slope(data, sample_idx=[0, 4]),
        bp.saliva.max_value(data),
        bp.saliva.initial_value(data),
    ]
    cort_feat = pd.concat(cort_feat, axis=1)
    return bp.saliva.utils.saliva_feature_wide_to_long(cort_feat, saliva_type="cortisol")


def compute_auc_increasing(data: pd.DataFrame, auc_type: str) -> pd.DataFrame:
    data = data.xs(auc_type, level="saliva_feature")
    data = data.unstack("reporting_type")["cortisol"].dropna()
    data = _columnwise_difference(data).round(2)
    data_increase = data > 0
    data_increase = data_increase.apply(lambda df: df.value_counts(normalize=True) * 100)
    data_increase = data_increase.reindex([True, False]).round(0)
    data_increase.index.name = "increasing"
    return data_increase


def _columnwise_difference(data: pd.DataFrame) -> pd.DataFrame:
    cols = data.columns
    df = data.values
    r, c = np.triu_indices(df.shape[1], 1)
    new_cols = [cols[i] + " | " + cols[j] for i, j in zip(r, c)]
    return pd.DataFrame(df[:, c] - df[:, r], columns=new_cols, index=data.index)
