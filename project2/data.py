import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


TARGET_COLUMN = "species"
CATEGORICAL_FEATURES = ["island", "sex"]
NUMERICAL_FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "year"]
DISPLAY_NUMERICAL_FEATURES = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
SPECIES_LABELS = ["Adelie", "Chinstrap", "Gentoo"]


def load_penguins_dataframe():
    try:
        from palmerpenguins import load_penguins

        df = load_penguins()
    except ImportError:
        try:
            import seaborn as sns

            df = sns.load_dataset("penguins")
        except ImportError:
            raise ImportError(
                "Install palmerpenguins or seaborn to load the Palmer Penguins dataset."
            )

    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str)
    return df.reset_index(drop=True)


def build_preprocessor():
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
            ("numerical", numerical_pipeline, NUMERICAL_FEATURES),
        ]
    )


def prepare_dataset(test_size=0.25, random_state=42):
    df = load_penguins_dataframe()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return {
        "dataframe": df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_columns": FEATURE_COLUMNS,
        "numerical_features": NUMERICAL_FEATURES,
        "display_numerical_features": DISPLAY_NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target_column": TARGET_COLUMN,
        "class_labels": sorted(y.unique().tolist()),
    }
