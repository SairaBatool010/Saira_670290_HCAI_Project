import os

from django.conf import settings

RANDOM_STATE = 42
VAL_SIZE = 0.1

CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
NUM_CLASSES = 4

ARTIFACTS_ROOT = os.path.join(settings.MEDIA_ROOT, "project3", "experiments")
TASK1_DIR = os.path.join(ARTIFACTS_ROOT, "task1")
TASK2_DIR = os.path.join(ARTIFACTS_ROOT, "task2")
TASK3_DIR = os.path.join(ARTIFACTS_ROOT, "task3")
TASK4_DIR = os.path.join(ARTIFACTS_ROOT, "task4")

# Expert B: a second simulated expert, deliberately weaker than the AI overall but
# genuinely stronger than the AI in one specific class (Business), so Tasks 3-4 can
# also demonstrate the case where deferral has real value, not just the case where
# it doesn't (Expert A).
TASK2B_DIR = os.path.join(ARTIFACTS_ROOT, "task2b")
TASK3B_DIR = os.path.join(ARTIFACTS_ROOT, "task3b")
TASK4B_DIR = os.path.join(ARTIFACTS_ROOT, "task4b")

AL_QUERY_BUDGETS = [20, 50, 100, 200, 500]
AL_BATCH_SIZE = 20

# Target per-class accuracy profile for simulated expert (calibrated on validation).
EXPERT_TARGET_CLASS_ACCURACY = {
    0: 0.74,  # World — moderate
    1: 0.90,  # Sports — very strong
    2: 0.86,  # Business — strong
    3: 0.58,  # Sci/Tech — weak (often confuses with Business)
}

# Expert B's target profile: weak everywhere except Business, where it is targeted to
# exceed the AI classifier's own Business accuracy (a specific, identifiable region of
# the input space where this expert -- unlike Expert A -- genuinely complements the AI).
EXPERT_B_TARGET_CLASS_ACCURACY = {
    0: 0.35,  # World — weak
    1: 0.35,  # Sports — weak
    2: 0.97,  # Business — specialist, aimed above the AI's own Business accuracy
    3: 0.35,  # Sci/Tech — weak
}

TFIDF_PARAM_GRID = {
    "vectorizer__ngram_range": [(1, 1), (1, 2)],
    "vectorizer__min_df": [2],
    "vectorizer__max_features": [10000, 20000],
    "vectorizer__sublinear_tf": [True],
    "vectorizer__stop_words": ["english"],
}

LR_PARAM_GRID = {
    "classifier__C": [0.1, 1.0, 10.0],
}

BASELINE_PARAM_GRID = {**TFIDF_PARAM_GRID, **LR_PARAM_GRID}
