import numpy as np
import pandas as pd


def _mad(values):
    median = np.median(values)
    return np.median(np.abs(values - median)) or 1.0


def _build_mad_weights(df, feature_columns, categorical_features):
    weights = {}
    for feature in feature_columns:
        if feature in categorical_features:
            weights[feature] = 1.0
        else:
            weights[feature] = _mad(df[feature].astype(float).values)
    return weights


def _mad_weighted_l1_batch(row, candidates, feature_columns, weights, categorical_features):
    distance = np.zeros(len(candidates))
    for feature in feature_columns:
        if feature in categorical_features:
            distance += np.where(candidates[feature].values == row[feature], 0.0, weights[feature])
        else:
            distance += np.abs(candidates[feature].values.astype(float) - float(row[feature])) / weights[feature]
    return distance


def _sample_candidate_batch(row, feature_columns, categorical_features, numerical_features, category_choices, size, spread_factor, categorical_change_prob, rng):
    candidates = {}
    for feature in numerical_features:
        spread = max(abs(float(row[feature])) * 0.08, 0.5) * spread_factor
        candidates[feature] = float(row[feature]) + rng.normal(0.0, spread, size=size)

    for feature in categorical_features:
        options = [value for value in category_choices[feature] if value != row[feature]]
        values = np.full(size, row[feature], dtype=object)
        if options:
            change_mask = rng.random(size) < categorical_change_prob
            replacements = rng.choice(options, size=change_mask.sum())
            values[change_mask] = replacements
        candidates[feature] = values

    return pd.DataFrame(candidates, columns=feature_columns)


def generate_counterfactuals(
    dataset,
    model_pipeline,
    instance_index,
    target_label,
    num_samples=1500,
    top_k=5,
    random_state=42,
):
    df = dataset["dataframe"]
    feature_columns = dataset["feature_columns"]
    categorical_features = dataset["categorical_features"]
    numerical_features = dataset["numerical_features"]
    row = df.loc[instance_index, feature_columns]
    weights = _build_mad_weights(df, feature_columns, categorical_features)
    rng = np.random.default_rng(random_state)

    category_choices = {
        feature: df[feature].dropna().unique().tolist() for feature in categorical_features
    }

    found = []
    attempts = 0
    max_attempts = 4
    current_samples = num_samples

    while not found and attempts < max_attempts:
        candidates = _sample_candidate_batch(
            row,
            feature_columns,
            categorical_features,
            numerical_features,
            category_choices,
            size=current_samples,
            spread_factor=1 + attempts * 0.5,
            categorical_change_prob=0.4 + 0.1 * attempts,
            rng=rng,
        )

        predictions = model_pipeline.predict(candidates)
        match_mask = predictions.astype(str) == str(target_label)

        if match_mask.any():
            matched_candidates = candidates.loc[match_mask].reset_index(drop=True)
            matched_predictions = predictions[match_mask]
            distances = _mad_weighted_l1_batch(
                row, matched_candidates, feature_columns, weights, categorical_features
            )
            for i in range(len(matched_candidates)):
                found.append(
                    {
                        "features": matched_candidates.iloc[i].to_dict(),
                        "prediction": str(matched_predictions[i]),
                        "distance": float(distances[i]),
                    }
                )

        attempts += 1
        current_samples = int(current_samples * 1.5)

    found.sort(key=lambda item: item["distance"])
    unique = []
    seen = set()
    for item in found:
        key = tuple(sorted(item["features"].items()))
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
        if len(unique) >= top_k:
            break

    return {
        "original": row.to_dict(),
        "original_prediction": str(model_pipeline.predict(row.to_frame().T)[0]),
        "target_label": str(target_label),
        "counterfactuals": unique,
        "searched_samples": current_samples,
    }
