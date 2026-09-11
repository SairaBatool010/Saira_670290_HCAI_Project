from django import forms

from .training import (
    CLASSIFICATION_METRICS,
    CLASSIFICATION_MODELS,
    REGRESSION_METRICS,
    REGRESSION_MODELS,
)


class CSVUploadForm(forms.Form):
    file = forms.FileField(label="Select a CSV file")


class PlotForm(forms.Form):
    feature_x = forms.ChoiceField(label="Feature X")
    feature_y = forms.ChoiceField(label="Feature Y")

    def __init__(self, *args, feature_names=None, **kwargs):
        super().__init__(*args, **kwargs)
        if feature_names:
            choices = [(name, name) for name in feature_names]
            self.fields["feature_x"].choices = choices
            self.fields["feature_y"].choices = choices


class TrainForm(forms.Form):
    model = forms.ChoiceField(label="Machine learning model")
    test_size = forms.FloatField(
        label="Test set size",
        min_value=0.1,
        max_value=0.5,
        initial=0.2,
        help_text="Fraction of the dataset reserved for testing (e.g. 0.2 = 20%).",
    )
    hyperparameter_values = forms.CharField(
        label="Hyperparameter values",
        required=False,
        help_text="Comma-separated values to try, e.g. 1,10,100",
    )
    metric = forms.ChoiceField(label="Evaluation metric")

    def __init__(self, *args, problem_type="classification", **kwargs):
        super().__init__(*args, **kwargs)
        if problem_type == "classification":
            model_choices = [(key, config["label"]) for key, config in CLASSIFICATION_MODELS.items()]
            metric_choices = [(key, label) for key, (label, _, _) in CLASSIFICATION_METRICS.items()]
            default_model = "logistic_regression"
            default_metric = "accuracy"
            default_values = CLASSIFICATION_MODELS[default_model]["default_values"]
        else:
            model_choices = [(key, config["label"]) for key, config in REGRESSION_MODELS.items()]
            metric_choices = [(key, label) for key, (label, _, _) in REGRESSION_METRICS.items()]
            default_model = "linear_regression"
            default_metric = "r2"
            default_values = REGRESSION_MODELS[default_model]["default_values"]

        self.fields["model"].choices = model_choices
        self.fields["metric"].choices = metric_choices
        if not self.is_bound:
            self.fields["model"].initial = default_model
            self.fields["metric"].initial = default_metric
            self.fields["hyperparameter_values"].initial = default_values
