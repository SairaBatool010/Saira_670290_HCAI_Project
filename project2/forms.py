from django import forms

from .data import DISPLAY_NUMERICAL_FEATURES
from .ml_pipeline import LAMBDA_MAX


class ExplainabilityForm(forms.Form):
    MODEL_CHOICES = [
        ("tree", "Decision tree"),
        ("logistic", "Logistic regression"),
    ]

    model_type = forms.ChoiceField(label="Model type", choices=MODEL_CHOICES, initial="tree")
    lambda_value = forms.FloatField(
        label="Regularization trade-off (lambda)",
        min_value=0.0,
        initial=0.0,
    )
    instance_index = forms.IntegerField(
        label="Example index",
        min_value=0,
        initial=0,
        help_text="Row number of the penguin to explain, from the dataset preview below (e.g. 0 for the first row).",
    )
    target_label = forms.ChoiceField(label="Target species")
    effect_feature = forms.ChoiceField(
        label="Numerical feature for effect plots",
        choices=[(feature, feature) for feature in DISPLAY_NUMERICAL_FEATURES],
        initial=DISPLAY_NUMERICAL_FEATURES[0],
    )

    def __init__(self, *args, class_labels=None, max_index=0, **kwargs):
        super().__init__(*args, **kwargs)
        if class_labels:
            self.fields["target_label"].choices = [(label, label) for label in class_labels]
            if not self.is_bound:
                self.fields["target_label"].initial = class_labels[0]
        self.fields["instance_index"].max_value = max_index
        self.fields["instance_index"].help_text = (
            f"Row number of the penguin to explain, from 0 to {max_index} (see the dataset "
            "preview below for example rows)."
        )

        model_type = (self.data.get("model_type") if self.is_bound else None) or "tree"
        self.fields["lambda_value"].max_value = LAMBDA_MAX.get(model_type, 0.05)
