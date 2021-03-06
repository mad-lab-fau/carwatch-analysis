from typing import Dict, Optional

import pandas as pd

from biopsykit.stats import StatsPipeline


def analysis_saliva_raw(data: pd.DataFrame, variable: str) -> StatsPipeline:
    pipeline = StatsPipeline(
        steps=[
            ('prep', 'normality'),
            ('prep', 'equal_var'),
            ('test', 'mixed_anova'),
            ('posthoc', 'pairwise_ttests')
        ],
        params={
            'prep__groupby': 'sample',
            'prep__group': variable,
            'dv': 'cortisol',
            'between': variable,
            'within': 'sample',
            'subject': 'night_id',
            'padjust': 'fdr_bh'
        }
    )

    pipeline.apply(data)
    return pipeline


def analysis_saliva_features(data: pd.DataFrame, variable: str,
                             test_type: Optional[str] = 'welch_anova') -> StatsPipeline:
    if test_type == 'welch_anova':
        posthoc = 'pairwise_tukey'
    else:
        posthoc = 'pairwise_ttests'

    steps = [
        ('prep', 'normality'),
        ('prep', 'equal_var'),
        ('test', test_type),
        ('posthoc', posthoc)
    ]
    params = {
        'groupby': 'biomarker',
        'group': variable,
        'dv': 'cortisol',
        'between': variable,
        'padjust': 'fdr_bh'
    }
    pipeline = StatsPipeline(
        steps=steps,
        params=params
    )

    pipeline.apply(data)
    return pipeline
