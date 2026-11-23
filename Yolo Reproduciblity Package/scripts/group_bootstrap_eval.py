"""
Paired Group-Bootstrap Resampling (Table 4 Replication).
Resamples 132 test groups with replacement 1,000 times.
Calculates 95% percentile bootstrap intervals for YOLO11n-seg minus YOLO26n-seg
and the empirical share of resamples where YOLO11n-seg outperforms YOLO26n-seg.
"""

import numpy as np
import pandas as pd

def compute_group_bootstrap_intervals(group_scores_df, n_resamples=1000, random_seed=42):
    """
    Args:
        group_scores_df: DataFrame indexed by group_id with columns:
                         ['yolo11_score', 'yolo26_score']
    """
    rng = np.random.RandomState(random_seed)
    unique_groups = group_scores_df.index.unique().values
    n_groups = len(unique_groups)

    diffs = []
    for _ in range(n_resamples):
        sample_groups = rng.choice(unique_groups, size=n_groups, replace=True)
        sample_df = group_scores_df.loc[sample_groups]
        mean_y11 = sample_df['yolo11_score'].mean()
        mean_y26 = sample_df['yolo26_score'].mean()
        diffs.append(mean_y11 - mean_y26)

    diffs = np.array(diffs)
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)
    mean_diff = np.mean(diffs)
    share_y11_higher = np.mean(diffs > 0)

    return {
        "mean_diff": mean_diff,
        "ci_95": (ci_lower, ci_upper),
        "share_higher": share_y11_higher
    }

if __name__ == "__main__":
    print("Paired Group-Bootstrap Evaluation Module")
    print("Used to generate 95% percentile intervals across 132 test groups.")
