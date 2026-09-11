import os
import uuid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from django.conf import settings
from django.shortcuts import redirect, render

from .forms import CSVUploadForm, PlotForm, TrainForm


def _save_uploaded_file(request, uploaded_file):
    upload_dir = os.path.join(settings.MEDIA_ROOT, "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    # Remove this session's previous upload so files don't pile up and so
    # concurrent sessions never collide on the same filename.
    old_path = request.session.get("csv_path")
    if old_path and os.path.exists(old_path) and os.path.dirname(old_path) == upload_dir:
        try:
            os.remove(old_path)
        except OSError:
            pass

    if not request.session.session_key:
        request.session.create()
    unique_name = f"{request.session.session_key}_{uuid.uuid4().hex[:8]}_{uploaded_file.name}"
    filepath = os.path.join(upload_dir, unique_name)
    with open(filepath, "wb+") as destination:
        for chunk in uploaded_file.chunks():
            destination.write(chunk)
    return filepath


ID_COLUMN_NAMES = {"id", "index", "unnamed: 0"}


def _drop_id_column(df):
    """Drop a leading identifier column, if the dataset appears to have one."""
    first_col = df.columns[0]
    looks_like_id = (
        first_col.strip().lower() in ID_COLUMN_NAMES
        or first_col.strip().lower().endswith("_id")
    )
    if looks_like_id and len(df.columns) > 2:
        return df.drop(columns=[first_col]), first_col
    return df, None


def _load_dataset(request):
    csv_path = request.session.get("csv_path")
    if not csv_path or not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    df, _ = _drop_id_column(df)
    return df


def _get_numeric_features(df, feature_names):
    return [name for name in feature_names if pd.api.types.is_numeric_dtype(df[name])]


def _detect_problem_type(label_series):
    if label_series.dtype == "object" or label_series.nunique() <= 10:
        return "classification"
    return "regression"


def _problem_type_reason(label_series):
    """Explain, in plain language, why the label column was interpreted this way."""
    n_unique = label_series.nunique()
    if label_series.dtype == "object":
        return (
            f'The label column contains text values ({n_unique} distinct categories), '
            "so it is treated as a classification problem."
        )
    if n_unique <= 10:
        return (
            f"The label column is numeric but only has {n_unique} distinct value(s) "
            "(10 or fewer), which usually indicates discrete categories encoded as numbers "
            "(e.g. 0/1/2 for species) rather than a continuous quantity, so it is treated as "
            "a classification problem."
        )
    return (
        f"The label column is numeric with {n_unique} distinct values (more than 10), "
        "consistent with a continuous quantity, so it is treated as a regression problem."
    )


def save_scatter_plot(x, y, labels, x_label, y_label, filename):
    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
    image_path = os.path.join(settings.MEDIA_ROOT, filename)

    plt.figure(figsize=(8, 6))
    if labels is not None:
        codes, unique_labels = pd.factorize(labels)
        plt.scatter(x, y, c=codes, cmap="viridis", alpha=0.8)
        for index, label in enumerate(unique_labels):
            color = plt.cm.viridis(index / max(len(unique_labels) - 1, 1))
            plt.scatter([], [], c=[color], label=str(label))
        plt.legend(title="Class")
    else:
        plt.scatter(x, y, alpha=0.8)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title("Data visualization")
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()
    return settings.MEDIA_URL + filename


def save_correlation_heatmap(df, numeric_features, filename):
    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
    image_path = os.path.join(settings.MEDIA_ROOT, filename)

    correlation = df[numeric_features].corr()
    plt.figure(figsize=(max(6, len(numeric_features) * 0.9), max(5, len(numeric_features) * 0.8)))
    plt.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    positions = range(len(numeric_features))
    plt.xticks(positions, numeric_features, rotation=45, ha="right")
    plt.yticks(positions, numeric_features)
    for i in range(len(numeric_features)):
        for j in range(len(numeric_features)):
            plt.text(
                j, i, f"{correlation.iloc[i, j]:.2f}", ha="center", va="center",
                color="white" if abs(correlation.iloc[i, j]) > 0.5 else "black", fontsize=8,
            )
    plt.title("Feature correlation heatmap")
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()
    return settings.MEDIA_URL + filename


def save_label_distribution_plot(df, label_name, problem_type, filename):
    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
    image_path = os.path.join(settings.MEDIA_ROOT, filename)

    plt.figure(figsize=(7, 5))
    if problem_type == "classification":
        counts = df[label_name].value_counts().sort_index()
        plt.bar(counts.index.astype(str), counts.values, color="#4C72B0")
        plt.ylabel("Number of rows")
        plt.title(f"Class balance for '{label_name}'")
    else:
        plt.hist(df[label_name].dropna(), bins=20, color="#4C72B0", edgecolor="white")
        plt.ylabel("Number of rows")
        plt.title(f"Distribution of '{label_name}'")
    plt.xlabel(label_name)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()
    return settings.MEDIA_URL + filename


def _build_visualization_context(request, df, feature_x, feature_y):
    feature_names = list(df.columns[:-1])
    label_name = df.columns[-1]
    numeric_features = _get_numeric_features(df, feature_names)
    non_numeric_features = [name for name in feature_names if name not in numeric_features]
    problem_type = _detect_problem_type(df[label_name])

    if feature_x not in numeric_features:
        feature_x = numeric_features[0]
    if feature_y not in numeric_features:
        feature_y = numeric_features[1 if len(numeric_features) > 1 else 0]

    single_feature_only = len(numeric_features) == 1
    if single_feature_only:
        # Only one numeric predictor available: plot it against the label instead.
        y_values = df[label_name]
        y_label = label_name
    else:
        y_values = df[feature_y]
        y_label = feature_y

    image_url = save_scatter_plot(
        df[feature_x],
        y_values,
        df[label_name] if problem_type == "classification" else None,
        feature_x,
        y_label,
        "project1_plot.png",
    )

    correlation_url = None
    if len(numeric_features) >= 2:
        correlation_url = save_correlation_heatmap(df, numeric_features, "project1_correlation.png")

    distribution_url = save_label_distribution_plot(
        df, label_name, problem_type, "project1_label_distribution.png"
    )

    plot_form = PlotForm(
        feature_names=numeric_features,
        initial={"feature_x": feature_x, "feature_y": feature_y},
    )

    return {
        "columns": list(df.columns),
        "feature_names": feature_names,
        "numeric_features": numeric_features,
        "non_numeric_features": non_numeric_features,
        "label_name": label_name,
        "problem_type": problem_type,
        "problem_type_reason": _problem_type_reason(df[label_name]),
        "row_count": len(df),
        "column_count": len(df.columns),
        "preview": df.head(10).to_html(classes="data-preview", index=False),
        "plot_form": plot_form,
        "image_url": image_url,
        "correlation_url": correlation_url,
        "distribution_url": distribution_url,
        "selected_x": feature_x,
        "selected_y": feature_y,
        "single_feature_only": single_feature_only,
        "dropped_id_column": request.session.get("dropped_id_column"),
    }


def index(request):
    form = CSVUploadForm()
    return render(request, "project1/index.html", {"form": form})


def upload_csv(request):
    if request.method != "POST":
        return redirect("project1:index")

    form = CSVUploadForm(request.POST, request.FILES)
    if not form.is_valid():
        return render(
            request,
            "project1/index.html",
            {"form": form, "error": "Please select a valid CSV file."},
        )

    uploaded_file = request.FILES["file"]
    if not uploaded_file.name.lower().endswith(".csv"):
        return render(
            request,
            "project1/index.html",
            {"form": form, "error": "Please upload a file with the .csv extension."},
        )

    try:
        try:
            df = pd.read_csv(uploaded_file)
        except (pd.errors.ParserError, UnicodeDecodeError):
            raise ValueError("The file could not be parsed as CSV. Please check its format and encoding.")

        if df.empty or len(df.columns) < 2:
            raise ValueError("The CSV must contain at least one feature column and one label column.")

        df, dropped_id_column = _drop_id_column(df)
        if len(df.columns) < 2:
            raise ValueError("The CSV must contain at least one feature column and one label column.")

        numeric_features = _get_numeric_features(df, list(df.columns[:-1]))
        if len(numeric_features) < 1:
            raise ValueError("At least one numeric feature column is required.")

        csv_path = _save_uploaded_file(request, uploaded_file)
        request.session["csv_path"] = csv_path
        request.session["csv_columns"] = list(df.columns)
        request.session["problem_type"] = _detect_problem_type(df[df.columns[-1]])
        request.session["dropped_id_column"] = dropped_id_column

        feature_y = numeric_features[1] if len(numeric_features) > 1 else numeric_features[0]
        context = _build_visualization_context(
            request,
            df,
            numeric_features[0],
            feature_y,
        )
        context["dropped_id_column"] = dropped_id_column
        return render(request, "project1/visualize.html", context)
    except ValueError as exc:
        return render(
            request,
            "project1/index.html",
            {"form": CSVUploadForm(), "error": str(exc)},
        )
    except Exception:
        return render(
            request,
            "project1/index.html",
            {"form": CSVUploadForm(), "error": "Could not process this file. Please upload a valid CSV."},
        )


def visualize(request):
    df = _load_dataset(request)
    if df is None:
        return redirect("project1:index")

    feature_names = list(df.columns[:-1])
    numeric_features = _get_numeric_features(df, feature_names)
    if len(numeric_features) < 1:
        return render(
            request,
            "project1/index.html",
            {
                "form": CSVUploadForm(),
                "error": "The uploaded dataset does not contain any numeric features.",
            },
        )

    if request.method == "POST" and len(numeric_features) > 1:
        plot_form = PlotForm(request.POST, feature_names=numeric_features)
        if plot_form.is_valid():
            feature_x = plot_form.cleaned_data["feature_x"]
            feature_y = plot_form.cleaned_data["feature_y"]
            context = _build_visualization_context(request, df, feature_x, feature_y)
            context["plot_form"] = plot_form
            return render(request, "project1/visualize.html", context)

    context = _build_visualization_context(
        request,
        df,
        numeric_features[0],
        numeric_features[1],
    )
    return render(request, "project1/visualize.html", context)


def _build_train_context(request, df, train_form=None, training_result=None, error=None):
    label_name = df.columns[-1]
    problem_type = _detect_problem_type(df[label_name])
    feature_count = len(_get_numeric_features(df, list(df.columns[:-1])))

    if train_form is None:
        train_form = TrainForm(problem_type=problem_type)

    from .training import CLASSIFICATION_MODELS, REGRESSION_MODELS

    models = CLASSIFICATION_MODELS if problem_type == "classification" else REGRESSION_MODELS
    model_param_info = {
        key: {
            "param": config["param"],
            "param_label": config["param_label"],
            "description": config["param_description"],
            "default_values": config["default_values"],
        }
        for key, config in models.items()
    }

    return {
        "label_name": label_name,
        "problem_type": problem_type,
        "sample_count": len(df),
        "feature_count": feature_count,
        "train_form": train_form,
        "training_result": training_result,
        "error": error,
        "model_param_info": model_param_info,
    }


def train(request):
    df = _load_dataset(request)
    if df is None:
        return redirect("project1:index")

    label_name = df.columns[-1]
    problem_type = _detect_problem_type(df[label_name])

    if request.method == "POST":
        train_form = TrainForm(request.POST, problem_type=problem_type)
        if train_form.is_valid():
            try:
                from .training import run_training

                training_result = run_training(
                    df,
                    model_key=train_form.cleaned_data["model"],
                    test_size=train_form.cleaned_data["test_size"],
                    hyperparameter_values=train_form.cleaned_data["hyperparameter_values"],
                    metric_key=train_form.cleaned_data["metric"],
                )
                context = _build_train_context(
                    request,
                    df,
                    train_form=train_form,
                    training_result=training_result,
                )
                return render(request, "project1/train.html", context)
            except ValueError as exc:
                context = _build_train_context(
                    request,
                    df,
                    train_form=train_form,
                    error=str(exc),
                )
                return render(request, "project1/train.html", context)
            except Exception:
                context = _build_train_context(
                    request,
                    df,
                    train_form=train_form,
                    error="Training failed unexpectedly. Please check your data and settings and try again.",
                )
                return render(request, "project1/train.html", context)

    context = _build_train_context(request, df)
    return render(request, "project1/train.html", context)
