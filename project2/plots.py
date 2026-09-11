import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from django.conf import settings
from sklearn.tree import plot_tree


def _media_path(filename):
    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
    return os.path.join(settings.MEDIA_ROOT, filename)


def _media_url(filename):
    return settings.MEDIA_URL + filename


def save_tree_plot(model_pipeline, class_labels, filename="project2_tree.png"):
    classifier = model_pipeline.named_steps["classifier"]
    preprocessor = model_pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()

    figure, axis = plt.subplots(figsize=(16, 8))
    plot_tree(
        classifier,
        feature_names=feature_names,
        class_names=class_labels,
        filled=True,
        rounded=True,
        fontsize=8,
        ax=axis,
    )
    axis.set_title("Decision tree structure")
    figure.tight_layout()
    image_path = _media_path(filename)
    figure.savefig(image_path, bbox_inches="tight")
    plt.close(figure)
    return _media_url(filename)


def save_probability_plot(x_values, curves, x_label, title, filename):
    figure, axis = plt.subplots(figsize=(9, 5))
    for label, y_values in curves.items():
        axis.plot(x_values, y_values, marker="o", linewidth=2, label=label)
    axis.set_xlabel(x_label)
    axis.set_ylabel("Class probability")
    axis.set_title(title)
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    image_path = _media_path(filename)
    figure.savefig(image_path, bbox_inches="tight")
    plt.close(figure)
    return _media_url(filename)


def save_feature_effect_plots(feature_name, pdp_data, ale_data, ale_method, filename_prefix):
    pdp_x, pdp_curves = pdp_data
    ale_x, ale_curves, _ = ale_data
    pdp_url = save_probability_plot(
        pdp_x,
        pdp_curves,
        feature_name,
        f"PDP for {feature_name}",
        f"{filename_prefix}_pdp.png",
    )
    ale_url = save_probability_plot(
        ale_x,
        ale_curves,
        feature_name,
        f"ALE for {feature_name} ({ale_method})",
        f"{filename_prefix}_ale.png",
    )
    return pdp_url, ale_url
