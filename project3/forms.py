from django import forms


class HumanExpertForm(forms.Form):
    LABEL_CHOICES = [
        (0, "World"),
        (1, "Sports"),
        (2, "Business"),
        (3, "Sci/Tech"),
    ]

    article_index = forms.IntegerField(
        label="Test article index (0–7599)",
        min_value=0,
        max_value=7599,
        initial=0,
    )
    expert_label = forms.ChoiceField(label="Your expert label", choices=LABEL_CHOICES)
