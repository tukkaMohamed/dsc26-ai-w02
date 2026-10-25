"""Data audit for Rawaj Pay transfer extracts."""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import roc_auc_score


def audit(df: pd.DataFrame, *, target: str, id_col: str, time_col: str) -> dict:
    # 1. Check required columns
    required = [target, id_col, time_col]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    # 2. Count rows
    n_rows = len(df)

    # 3. Identify missing values, including blank strings
    def is_missing(series):
        return series.isna() | series.map(
            lambda value: isinstance(value, str) and value.strip() == ""
        )

    missing_masks = {
        col: is_missing(df[col])
        for col in df.columns
    }

    missing = {
        col: float(mask.mean()) if n_rows else 0.0
        for col, mask in missing_masks.items()
    }

    # 4. Find duplicate IDs, ignoring missing IDs
    valid_ids = df.loc[~missing_masks[id_col], id_col]
    duplicate_ids = valid_ids[
        valid_ids.duplicated(keep=False)
    ].unique().tolist()
    duplicate_ids = sorted(duplicate_ids, key=str)

    # 5. Count rows that exactly repeat an earlier row
    duplicate_rows = int(df.duplicated().sum())

    # 6. Find constant columns, excluding the target
    constant_columns = []
    for col in df.columns:
        if col == target:
            continue

        values = df.loc[~missing_masks[col], col]
        if values.nunique(dropna=True) <= 1:
            constant_columns.append(col)

    constant_columns.sort()

    # 7. Count rows without labels
    target_missing = missing_masks[target]
    labelled_mask = ~target_missing
    labelled = df.loc[labelled_mask]
    y = labelled[target]

    unlabelled_rows = int(target_missing.sum())

    # 8. Calculate positive rate
    positive_rate = float((y == 1).mean()) if len(y) else 0.0

    # 9. Check timestamps
    time_missing = missing_masks[time_col]
    parsed = pd.to_datetime(
        df[time_col].mask(time_missing),
        errors="coerce",
    )

    bad_timestamps = int(
        ((~time_missing) & parsed.isna()).sum()
    )

    valid_times = parsed.dropna()

    if len(valid_times):
        time_range = [
            valid_times.min().isoformat(),
            valid_times.max().isoformat(),
        ]
    else:
        time_range = [None, None]

    # 10. Find possible data leakage
    leakage_suspects = []

    if len(labelled) >= 20 and y.nunique() == 2:
        for col in df.columns:
            if col in {target, id_col, time_col}:
                continue

            series = labelled[col]
            miss = missing_masks[col].loc[labelled_mask]

            # Count missing as a distinct value
            distinct_count = (
                series.nunique(dropna=True) + int(miss.any())
            )

            # A constant column cannot be a leakage suspect
            if distinct_count <= 1:
                continue

            suspect = False

            # Check whether missingness strongly predicts the target
            missing_auc = roc_auc_score(y, miss.astype(int))

            if missing_auc >= 0.98 or missing_auc <= 0.02:
                suspect = True

            # Check numeric and boolean values
            is_numeric = (
                pd.api.types.is_numeric_dtype(series.dtype)
                or pd.api.types.is_bool_dtype(series.dtype)
            )

            if is_numeric:
                observed = ~miss

                if int(observed.sum()) >= 20:
                    observed_y = y.loc[observed]
                    observed_values = series.loc[observed]

                    if observed_y.nunique() == 2:
                        value_auc = roc_auc_score(
                            observed_y,
                            observed_values,
                        )

                        if value_auc >= 0.98 or value_auc <= 0.02:
                            suspect = True

            if suspect:
                leakage_suspects.append(col)

    leakage_suspects.sort()

    # 11. Return exactly the required keys
    return {
        "n_rows": int(n_rows),
        "missing": missing,
        "duplicate_ids": duplicate_ids,
        "duplicate_rows": duplicate_rows,
        "constant_columns": constant_columns,
        "unlabelled_rows": unlabelled_rows,
        "positive_rate": positive_rate,
        "bad_timestamps": bad_timestamps,
        "time_range": time_range,
        "leakage_suspects": leakage_suspects,
    }
