import numpy as np
from sklearn.model_selection import train_test_split

from .config import CLASS_NAMES, RANDOM_STATE, VAL_SIZE


def _load_raw_ag_news():
    try:
        from datasets import load_dataset

        dataset = load_dataset("ag_news")
    except Exception:
        from datasets import load_dataset

        dataset = load_dataset("fancyzhx/ag_news")

    train = dataset["train"]
    test = dataset["test"]

    X_train_full = [row["text"] for row in train]
    y_train_full = np.array([row["label"] for row in train], dtype=int)
    X_test = [row["text"] for row in test]
    y_test = np.array([row["label"] for row in test], dtype=int)
    return X_train_full, y_train_full, X_test, y_test


def load_ag_news_splits(random_state=RANDOM_STATE, val_size=VAL_SIZE):
    X_train_full, y_train_full, X_test, y_test = _load_raw_ag_news()

    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train_full,
    )

    return {
        "X_train_full": X_train_full,
        "y_train_full": y_train_full,
        "X_train_fit": X_train_fit,
        "y_train_fit": y_train_fit,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "class_names": CLASS_NAMES,
        "random_state": random_state,
        "val_size": val_size,
    }
