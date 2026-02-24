"""Stratified train/validation/test splitting utilities.

Provides language-aware and intent-aware stratified splitting to ensure
representative distributions across all data partitions.
"""

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def stratified_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify_col: str = "intent",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train/val/test sets with stratification.

    Performs a two-stage split: first train vs. remaining, then
    remaining into val and test, maintaining class distribution.

    Args:
        df: Input DataFrame to split.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        test_ratio: Fraction for test set.
        stratify_col: Column to stratify on.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).

    Raises:
        ValueError: If ratios don't sum to approximately 1.0.
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.3f}")

    stratify = df[stratify_col] if stratify_col in df.columns else None

    # Guard against stratification with too few samples per class
    if stratify is not None:
        min_count = stratify.value_counts().min()
        if min_count < 2:
            logger.warning(
                "Class with only %d sample(s) in '%s'; disabling stratification",
                min_count,
                stratify_col,
            )
            stratify = None

    remaining_ratio = val_ratio + test_ratio
    train_df, remaining_df = train_test_split(
        df,
        test_size=remaining_ratio,
        stratify=stratify,
        random_state=random_state,
    )

    if stratify is not None:
        remaining_stratify = remaining_df[stratify_col]
        min_remaining = remaining_stratify.value_counts().min()
        if min_remaining < 2:
            remaining_stratify = None
    else:
        remaining_stratify = None

    relative_test = test_ratio / remaining_ratio
    val_df, test_df = train_test_split(
        remaining_df,
        test_size=relative_test,
        stratify=remaining_stratify,
        random_state=random_state,
    )

    logger.info(
        "Split complete: train=%d, val=%d, test=%d",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    return train_df, val_df, test_df


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str = "data/processed",
) -> dict[str, Path]:
    """Save train/val/test splits to parquet files.

    Args:
        train_df: Training set DataFrame.
        val_df: Validation set DataFrame.
        test_df: Test set DataFrame.
        output_dir: Directory to save the split files.

    Returns:
        Dictionary mapping split names to their file paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = out / f"{name}.parquet"
        df.to_parquet(path, index=False)
        paths[name] = path
        logger.info("Saved %s split: %d rows -> %s", name, len(df), path)

    return paths
