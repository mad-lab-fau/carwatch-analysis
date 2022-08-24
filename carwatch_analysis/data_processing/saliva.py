import biopsykit as bp
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
