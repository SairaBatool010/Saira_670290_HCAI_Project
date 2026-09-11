import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _predict_proba(model_pipeline, X_frame):
    return model_pipeline.predict_proba(X_frame)


def compute_pdp(model_pipeline, dataset, feature_name, grid_size=30):
    df = dataset["dataframe"]
    feature_columns = dataset["feature_columns"]
    class_labels = model_pipeline.named_steps["classifier"].classes_
    grid = np.linspace(df[feature_name].min(), df[feature_name].max(), grid_size)
    curves = {label: [] for label in class_labels}

    for value in grid:
        modified = df[feature_columns].copy()
        modified[feature_name] = value
        probabilities = _predict_proba(model_pipeline, modified)
        for class_index, label in enumerate(class_labels):
            curves[label].append(float(probabilities[:, class_index].mean()))

    return grid.tolist(), curves


def _finite_difference_derivative_batch(model_pipeline, dataset, feature_name, sample_indices, step):
    feature_columns = dataset["feature_columns"]
    base_rows = dataset["dataframe"].loc[sample_indices, feature_columns].copy()
    plus_rows = base_rows.copy()
    minus_rows = base_rows.copy()
    plus_rows[feature_name] = base_rows[feature_name].astype(float) + step
    minus_rows[feature_name] = base_rows[feature_name].astype(float) - step

    plus_proba = _predict_proba(model_pipeline, plus_rows)
    minus_proba = _predict_proba(model_pipeline, minus_rows)
    return (plus_proba - minus_proba) / (2 * step)


def _exact_logistic_derivative_batch(model_pipeline, dataset, feature_name, sample_indices):
    preprocessor = model_pipeline.named_steps["preprocessor"]
    classifier = model_pipeline.named_steps["classifier"]
    if not isinstance(classifier, LogisticRegression):
        return None

    feature_columns = dataset["feature_columns"]
    rows = dataset["dataframe"].loc[sample_indices, feature_columns]
    transformed = preprocessor.transform(rows)
    coef = classifier.coef_
    intercept = classifier.intercept_

    transformed_array = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
    logits = transformed_array @ coef.T + intercept
    probabilities = softmax(logits)  # shape: (n_rows, n_classes)

    feature_names = preprocessor.get_feature_names_out()
    feature_index = None
    for index, name in enumerate(feature_names):
        if feature_name in name:
            feature_index = index
            break
    if feature_index is None:
        return None

    n_classes = len(classifier.classes_)
    coef_for_feature = coef[:, feature_index]  # shape: (n_classes,)
    # Jacobian of softmax w.r.t. this feature for every row at once:
    # d p_k / d x = p_k * (coef_k - sum_j p_j * coef_j)
    weighted_sum = probabilities @ coef_for_feature  # shape: (n_rows,)
    derivatives = probabilities * (coef_for_feature[None, :] - weighted_sum[:, None])
    return derivatives  # shape: (n_rows, n_classes)


def softmax(values):
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def compute_ale(model_pipeline, dataset, feature_name, bins=12):
    df = dataset["dataframe"]
    feature_columns = dataset["feature_columns"]
    class_labels = model_pipeline.named_steps["classifier"].classes_
    sorted_df = df.sort_values(feature_name).reset_index()
    bin_edges = np.quantile(sorted_df[feature_name], np.linspace(0, 1, bins + 1))
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 3:
        bin_edges = np.linspace(sorted_df[feature_name].min(), sorted_df[feature_name].max(), bins + 1)

    is_logistic = isinstance(model_pipeline.named_steps["classifier"], LogisticRegression)
    ale_curves = {label: [] for label in class_labels}
    bin_centers = []

    accumulated = np.zeros(len(class_labels))
    for bin_index in range(len(bin_edges) - 1):
        lower = bin_edges[bin_index]
        upper = bin_edges[bin_index + 1]
        mask = (sorted_df[feature_name] >= lower) & (sorted_df[feature_name] <= upper)
        interval_rows = sorted_df.loc[mask]
        if interval_rows.empty:
            continue

        # The finite-difference step is tied to the bin width itself (rather than a
        # tiny fixed epsilon) for non-differentiable models: a decision tree's
        # predict_proba is piecewise constant, so an infinitesimal step almost never
        # crosses a split threshold, which would make the derivative estimate zero
        # almost everywhere. Using a bin-scale step is the "discretization" needed
        # to estimate a derivative for a model that has none.
        discretized_step = max((upper - lower) / 2, 1e-6)
        sample_indices = interval_rows["index"].astype(int).tolist()

        derivatives = None
        if is_logistic:
            derivatives = _exact_logistic_derivative_batch(model_pipeline, dataset, feature_name, sample_indices)
        if derivatives is None:
            derivatives = _finite_difference_derivative_batch(
                model_pipeline, dataset, feature_name, sample_indices, step=discretized_step
            )

        local_effect_sum = derivatives.sum(axis=0) * (upper - lower)
        accumulated += local_effect_sum / len(interval_rows)
        bin_centers.append(float((lower + upper) / 2))
        for class_index, label in enumerate(class_labels):
            ale_curves[label].append(float(accumulated[class_index]))

    for label in class_labels:
        values = ale_curves[label]
        if values:
            mean_value = float(np.mean(values))
            ale_curves[label] = [value - mean_value for value in values]

    return bin_centers, ale_curves, "exact" if is_logistic else "finite differences"
