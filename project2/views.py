from django.shortcuts import render

from .counterfactuals import generate_counterfactuals
from .data import DISPLAY_NUMERICAL_FEATURES, prepare_dataset
from .feature_effects import compute_ale, compute_pdp
from .forms import ExplainabilityForm
from .ml_pipeline import LAMBDA_MAX, complexity_label, get_model_bank, select_model
from .plots import save_feature_effect_plots, save_tree_plot


_DATASET = None
_MODEL_CACHE = {}


def _get_dataset():
    global _DATASET
    if _DATASET is None:
        _DATASET = prepare_dataset()
    return _DATASET


def _get_models(model_type):
    if model_type not in _MODEL_CACHE:
        _MODEL_CACHE[model_type] = get_model_bank(_get_dataset(), model_type)
    return _MODEL_CACHE[model_type]


def _default_form_data(dataset):
    default_index = 0
    current_label = dataset["dataframe"].loc[default_index, dataset["target_column"]]
    other_labels = [label for label in dataset["class_labels"] if label != current_label]
    default_target = other_labels[0] if other_labels else dataset["class_labels"][0]

    return {
        "model_type": "tree",
        "lambda_value": "0.0",
        "instance_index": str(default_index),
        "target_label": default_target,
        "effect_feature": DISPLAY_NUMERICAL_FEATURES[0],
    }


def _build_context(form):
    dataset = _get_dataset()
    model_type = form.cleaned_data["model_type"]
    lambda_value = form.cleaned_data["lambda_value"]
    instance_index = form.cleaned_data["instance_index"]
    target_label = form.cleaned_data["target_label"]
    effect_feature = form.cleaned_data["effect_feature"]

    models = _get_models(model_type)
    selected_model, scored_models = select_model(models, lambda_value)
    pipeline = selected_model["pipeline"]
    class_labels = pipeline.named_steps["classifier"].classes_.tolist()

    # Task 1: a single, unconstrained decision tree fit with no regularization at all,
    # shown independently of the lambda-driven selection used for Tasks 2-3. It gets its
    # own tree diagram (not reused from the lambda-selected model below), so Task 1's
    # deliverable is always visible regardless of the slider or model_type chosen.
    baseline_tree = next(
        model for model in _get_models("tree") if model["parameter"] is None
    )
    baseline_class_labels = baseline_tree["pipeline"].named_steps["classifier"].classes_.tolist()
    baseline_tree_image_url = save_tree_plot(
        baseline_tree["pipeline"], baseline_class_labels, filename="project2_baseline_tree.png"
    )

    tree_image_url = None
    if model_type == "tree":
        tree_image_url = save_tree_plot(pipeline, class_labels)

    counterfactuals = generate_counterfactuals(
        dataset,
        pipeline,
        instance_index=instance_index,
        target_label=target_label,
    )
    counterfactuals["already_target_class"] = (
        counterfactuals["original_prediction"] == str(target_label)
    )

    def _format_value(value):
        return round(float(value), 2) if isinstance(value, (int, float)) else value

    feature_columns = dataset["feature_columns"]
    original_values = counterfactuals["original"]
    for item in counterfactuals["counterfactuals"]:
        item["feature_rows"] = []
        for feature in feature_columns:
            original_display = _format_value(original_values[feature])
            counterfactual_display = _format_value(item["features"][feature])
            item["feature_rows"].append(
                {
                    "feature": feature,
                    "original": original_display,
                    "counterfactual": counterfactual_display,
                    # Compare the rounded display values, not raw floats, so a change
                    # of a few hundredths (below the displayed precision) isn't
                    # highlighted as if it were a meaningful difference.
                    "changed": original_display != counterfactual_display,
                }
            )

    pdp_data = compute_pdp(pipeline, dataset, effect_feature)
    ale_data = compute_ale(pipeline, dataset, effect_feature)
    pdp_url, ale_url = save_feature_effect_plots(
        effect_feature,
        pdp_data,
        ale_data,
        ale_data[2],
        f"project2_{model_type}_{effect_feature}",
    )

    model_rows = []
    for model_info in scored_models:
        model_rows.append(
            {
                "setting": model_info["setting_label"],
                "accuracy": round(model_info["accuracy"], 4),
                "complexity": round(model_info["complexity"], 4),
                "selection_score": round(model_info["selection_score"], 4),
                "selected": model_info["id"] == selected_model["id"],
            }
        )

    return {
        "form": form,
        "dataset_size": len(dataset["dataframe"]),
        "class_labels": class_labels,
        "baseline_tree_accuracy": round(baseline_tree["accuracy"], 4),
        "baseline_tree_leaves": int(baseline_tree["complexity"]),
        "baseline_tree_image_url": baseline_tree_image_url,
        "selected_model": selected_model,
        "model_rows": model_rows,
        "complexity_label": complexity_label(model_type),
        "lambda_value": lambda_value,
        "lambda_max": LAMBDA_MAX.get(model_type, 0.05),
        "lambda_max_tree": LAMBDA_MAX["tree"],
        "lambda_max_logistic": LAMBDA_MAX["logistic"],
        "tree_image_url": tree_image_url,
        "counterfactuals": counterfactuals,
        "pdp_url": pdp_url,
        "ale_url": ale_url,
        "ale_method": ale_data[2],
        "effect_feature": effect_feature,
        "preview_rows": dataset["dataframe"].head(8).to_html(index=True, classes="data-preview"),
    }


def index(request):
    dataset = _get_dataset()
    max_index = len(dataset["dataframe"]) - 1
    form_data = request.POST if request.method == "POST" else _default_form_data(dataset)

    form = ExplainabilityForm(
        form_data,
        class_labels=dataset["class_labels"],
        max_index=max_index,
    )
    if form.is_valid():
        return render(request, "project2/index.html", _build_context(form))

    return render(
        request,
        "project2/index.html",
        {
            "form": form,
            "dataset_size": len(dataset["dataframe"]),
            "class_labels": dataset["class_labels"],
            "preview_rows": dataset["dataframe"].head(8).to_html(index=True, classes="data-preview"),
            "error": "Please check the form inputs.",
        },
    )
