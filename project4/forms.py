from django import forms


class StartStudyForm(forms.Form):
    DESIGN_CHOICES = [
        ("pairwise", "Design 1: Pairwise comparisons"),
        ("ranking", "Design 2: Rank 10 movies"),
        ("random", "Random assignment (recommended for study)"),
    ]

    participant_id = forms.CharField(label="Participant ID", max_length=50)
    design = forms.ChoiceField(label="Study condition", choices=DESIGN_CHOICES, initial="random")


class ConsentForm(forms.Form):
    AGE_CHOICES = [
        ("18-24", "18-24"),
        ("25-34", "25-34"),
        ("35-44", "35-44"),
        ("45-54", "45-54"),
        ("55+", "55+"),
    ]
    FREQUENCY_CHOICES = [
        ("weekly", "At least weekly"),
        ("monthly", "A few times a month"),
        ("occasionally", "Occasionally"),
        ("rarely", "Rarely"),
    ]

    age_range = forms.ChoiceField(label="Age range", choices=AGE_CHOICES)
    movie_frequency = forms.ChoiceField(
        label="How often do you watch movies?", choices=FREQUENCY_CHOICES
    )
    consent = forms.BooleanField(
        label=(
            "I have read the study description, I am 18 or older, and I voluntarily agree "
            "to participate. I understand my responses are used only to demonstrate the "
            "elicitation interface and are not part of a real research study."
        ),
        required=True,
    )
