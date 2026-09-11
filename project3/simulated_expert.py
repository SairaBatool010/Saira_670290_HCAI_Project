"""
Phase 4 — Simulated human expert for Project 3.

The expert predicts from article text only. True labels are never passed to predict().
They are used only by the evaluation/calibration framework.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .config import CLASS_NAMES, EXPERT_TARGET_CLASS_ACCURACY, NUM_CLASSES, RANDOM_STATE

DOMAIN_KEYWORDS_EXPERT_A = {
    0: [
        "government",
        "war",
        "election",
        "president",
        "minister",
        "united nations",
        "parliament",
        "diplomat",
        "conflict",
        "terror",
    ],
    1: [
        "game",
        "team",
        "coach",
        "score",
        "championship",
        "player",
        "league",
        "match",
        "tournament",
        "olympic",
    ],
    2: [
        "market",
        "stock",
        "company",
        "profit",
        "economy",
        "shares",
        "investor",
        "revenue",
        "bank",
        "trade",
    ],
    3: [
        "software",
        "technology",
        "computer",
        "research",
        "internet",
        "microsoft",
        "digital",
        "chip",
        "wireless",
        "online",
    ],
}

# Off-diagonal mass reflects realistic confusions (Sci/Tech <-> Business, World <-> Business).
DEFAULT_CONFUSION_OFF_DIAGONAL = {
    0: {1: 0.08, 2: 0.12, 3: 0.06},
    1: {0: 0.04, 2: 0.04, 3: 0.02},
    2: {0: 0.05, 1: 0.03, 3: 0.06},
    3: {0: 0.04, 1: 0.02, 2: 0.34},
}

# Expert B: a Business specialist. No keyword lists at all for the other three classes
# (deliberately -- see note below), and a large, high-precision vocabulary of finance
# and markets terms for Business -- realistic domain jargon that reliably marks an
# article as Business even from a "weak" bag-of-words reader.
#
# Design note: an earlier version gave the other three classes small generic keyword
# lists (e.g. "government", "game", "computer"). That measurably hurt Business's own
# accuracy: those generic words occasionally appear in Business articles too (e.g. "the
# company's new computer chip"), which spuriously marked those rows as having keyword
# "evidence" for the wrong class and diluted the Business signal. Leaving the other
# classes with no keyword list at all means adaptive_keyword_weight (see below) falls
# back to the plain text pre-model for them, with no risk of a false-positive trigger.
DOMAIN_KEYWORDS_EXPERT_B = {
    0: [],
    1: [],
    2: [
        "market", "stock", "stocks", "shares", "share price", "nasdaq", "nyse",
        "dow jones", "wall street", "earnings", "quarterly", "profit", "profits",
        "revenue", "investor", "investors", "shareholder", "shareholders", "ipo",
        "merger", "acquisition", "ceo", "economy", "economic", "inflation",
        "interest rate", "interest rates", "federal reserve", "bond", "bonds",
        "trade", "trading", "hedge fund", "dividend", "dividends", "bankruptcy",
        "takeover", "billion", "million", "percent", "%", "$",
        # Broader business-desk vocabulary (retail, industry, labor, commodities).
        "sales", "sale", "forecast", "company", "companies", "corporate",
        "corporation", "retailer", "retail", "chain", "chains", "industry",
        "manufacturer", "manufacturers", "exports", "imports", "tariff",
        "tariffs", "consumer", "consumers", "jobs", "labor", "job cuts",
        "layoffs", "workforce", "airline", "airlines", "oil price", "oil prices",
        "crude oil", "gas prices", "supply", "demand", "deal", "contract",
        "contracts", "chief executive", "chairman", "outsourcing", "wal-mart",
        "prices rose", "prices fell", "stake", "venture",
        # Terms mined from train_fit by log-odds (business vs. other classes),
        # filtered to genuine vocabulary only -- dataset-specific scraping
        # artifacts (URL fragments, ticker symbols) were excluded on inspection
        # as not representing real domain expertise.
        "treasuries", "mortgage", "mortgages", "mortgage rates", "oil futures",
        "opec", "mutual fund", "mutual funds", "insurance broker", "quarterly profit",
        "corporate profits", "home sales", "store sales", "stocks fell", "stocks rose",
        "bankruptcy filing", "pilots union", "flight attendants", "finance company",
        "brokerage", "lender", "securities", "raise rates", "winter fuel",
        "drug maker", "pharmaceuticals", "holding corp", "high oil",
    ],
    3: [],
}

# Off-diagonal mass for Expert B: when wrong, the weak classes mostly confuse among
# themselves (not into Business), so Business's precision stays reasonable, while
# Business itself (when wrong) spreads its small error mass evenly.
EXPERT_B_CONFUSION_OFF_DIAGONAL = {
    0: {1: 0.45, 3: 0.55},
    1: {0: 0.45, 3: 0.55},
    2: {0: 0.34, 1: 0.33, 3: 0.33},
    3: {0: 0.55, 1: 0.45},
}


def build_confusion_matrix(target_accuracy, off_diagonal=None):
    if off_diagonal is None:
        off_diagonal = DEFAULT_CONFUSION_OFF_DIAGONAL

    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=float)
    for row in range(NUM_CLASSES):
        matrix[row, row] = target_accuracy[row]
        remaining = 1.0 - matrix[row, row]
        weights = off_diagonal[row]
        total_weight = sum(weights.values())
        for col, weight in weights.items():
            matrix[row, col] = remaining * (weight / total_weight)
    return matrix


class SimulatedExpert:
    """
    Domain-specialist journalist: strong on Sports/Business, moderate on World,
    weak on Sci/Tech. Predictions depend on text features and a calibrated
    class-conditional confusion matrix only.
    """

    name = "Domain specialist journalist"
    description = (
        "A simulated news editor with strong Sports and Business expertise, "
        "moderate World coverage, and weaker Sci/Tech judgment (often confuses "
        "technology articles with Business)."
    )

    def __init__(
        self,
        confusion_matrix,
        random_state=RANDOM_STATE,
        domain_keywords=None,
        keyword_weight=0.35,
        name=None,
        description=None,
        sharpness=1.0,
        adaptive_keyword_weight=False,
    ):
        self.confusion_matrix = np.asarray(confusion_matrix, dtype=float)
        self.rng = np.random.default_rng(random_state)
        # domain_keywords/keyword_weight default to Expert A's original values, so
        # existing behavior is unchanged unless a caller explicitly overrides them
        # (used by Expert B to specialize in one class).
        self.domain_keywords = domain_keywords if domain_keywords is not None else DOMAIN_KEYWORDS_EXPERT_A
        self.keyword_weight = keyword_weight
        # sharpness=1.0 leaves the belief distribution untouched (Expert A's exact
        # original behavior). A latent topic is drawn stochastically from the full
        # belief distribution, not just its argmax -- so even when a class is the
        # clear favorite, a diffuse (low-confidence) distribution lets the sampler
        # wander elsewhere far more often than the argmax alone would suggest.
        # sharpness > 1 raises the belief to a power before renormalizing, letting a
        # specialist expert's confident, keyword-driven signal actually dominate the
        # stochastic draw when the signal is present, while ambiguous cases (no
        # matching keywords) remain close to uniform and are still drawn randomly.
        self.sharpness = sharpness
        # When False (Expert A's original behavior), keyword_weight is applied
        # uniformly to every row, even rows where none of the domain keyword lists
        # matched anything -- on those rows keyword_proba is just a uniform prior,
        # and blending it in at a fixed weight dilutes whatever signal the text
        # pre-model had, rather than adding information. When True, the keyword
        # weight is only applied on rows where at least one keyword actually
        # matched; rows with no keyword evidence fall back to the pre-model alone.
        self.adaptive_keyword_weight = adaptive_keyword_weight
        if name is not None:
            self.name = name
        if description is not None:
            self.description = description
        self.pipeline = Pipeline(
            steps=[
                (
                    "vectorizer",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=2,
                        max_features=8000,
                        sublinear_tf=True,
                        stop_words="english",
                    ),
                ),
                ("classifier", MultinomialNB(alpha=0.5)),
            ]
        )

    def fit(self, texts, labels):
        self.pipeline.fit(texts, labels)
        return self

    def _keyword_scores(self, texts):
        scores = np.zeros((len(texts), NUM_CLASSES), dtype=float)
        for index, text in enumerate(texts):
            text_lower = text.lower()
            for label, keywords in self.domain_keywords.items():
                scores[index, label] = sum(1 for word in keywords if word in text_lower)
        return scores

    def _combined_proba(self, texts):
        model_proba = self.pipeline.predict_proba(texts)
        keyword_scores = self._keyword_scores(texts)
        keyword_proba = keyword_scores + 0.05
        keyword_proba = keyword_proba / keyword_proba.sum(axis=1, keepdims=True)

        if self.adaptive_keyword_weight:
            has_evidence = (keyword_scores.sum(axis=1) > 0).astype(float)
            row_weight = (self.keyword_weight * has_evidence)[:, None]
        else:
            row_weight = self.keyword_weight

        combined = (1 - row_weight) * model_proba + row_weight * keyword_proba
        combined = combined / combined.sum(axis=1, keepdims=True)
        if self.sharpness != 1.0:
            combined = combined ** self.sharpness
            combined = combined / combined.sum(axis=1, keepdims=True)
        return combined

    def precompute_features(self, texts):
        return self._combined_proba(texts)

    def predict_from_features(self, combined_proba):
        predictions = np.empty(len(combined_proba), dtype=int)
        for index, proba_row in enumerate(combined_proba):
            latent_topic = int(self.rng.choice(NUM_CLASSES, p=proba_row))
            predictions[index] = int(
                self.rng.choice(NUM_CLASSES, p=self.confusion_matrix[latent_topic])
            )
        return predictions

    def predict(self, texts):
        """Predict labels from text only. Must not receive true labels."""
        combined_proba = self.precompute_features(texts)
        return self.predict_from_features(combined_proba)

    def competence_profile(self):
        return {
            CLASS_NAMES[label]: round(float(self.confusion_matrix[label, label]), 3)
            for label in range(NUM_CLASSES)
        }


def _per_class_accuracy(y_true, y_pred):
    metrics = {}
    for label in range(NUM_CLASSES):
        mask = y_true == label
        if mask.any():
            metrics[label] = float((y_pred[mask] == y_true[mask]).mean())
        else:
            metrics[label] = 0.0
    return metrics


def pre_model_accuracy(combined_proba, y_true):
    """
    Standalone accuracy of the text pre-model + keyword blend alone (stage 1-2),
    isolated from the confusion-matrix draw (stage 3): argmax(combined_proba) vs the
    true label. This is the "P(latent topic == true label)" term that the confusion
    matrix's calibrated diagonal gets multiplied by, so it quantifies exactly how much
    of the gap between the target profile and the measured, post-sampling accuracy is
    attributable to the pre-model's own imperfection (see Task 2 of the report).
    """
    predictions = np.asarray(combined_proba).argmax(axis=1)
    return {
        "overall": float((predictions == y_true).mean()),
        "per_class": _per_class_accuracy(y_true, predictions),
    }


def _default_off_diagonal_variants():
    # Reproduces Expert A's original search space exactly: 5 variants of the Sci/Tech
    # row's off-diagonal weighting toward Business, evaluated in the same order.
    variants = []
    for sci_weight in [0.24, 0.28, 0.32, 0.36, 0.40]:
        off_diagonal = {
            key: value.copy() if isinstance(value, dict) else value
            for key, value in DEFAULT_CONFUSION_OFF_DIAGONAL.items()
        }
        off_diagonal[3] = {0: 0.04, 1: 0.02, 2: sci_weight}
        variants.append(off_diagonal)
    return variants


def _search_confusion_matrix(
    y_true,
    combined_proba,
    targets,
    off_diagonal_variants=None,
    clip_range=(0.45, 0.95),
    scales=(0.92, 0.96, 1.0, 1.04, 1.08),
    random_state=RANDOM_STATE,
):
    if off_diagonal_variants is None:
        off_diagonal_variants = _default_off_diagonal_variants()

    best_matrix = build_confusion_matrix(targets)
    best_score = -np.inf
    best_pred = None

    for off_diagonal in off_diagonal_variants:
        for scale in scales:
            candidate_targets = {
                label: float(np.clip(targets[label] * scale, *clip_range))
                for label in targets
            }
            matrix = build_confusion_matrix(candidate_targets, off_diagonal=off_diagonal)
            trial_expert = SimulatedExpert(matrix, random_state=random_state)
            val_pred = trial_expert.predict_from_features(combined_proba)
            per_class = _per_class_accuracy(y_true, val_pred)
            score = -sum((per_class[label] - targets[label]) ** 2 for label in targets)
            if score > best_score:
                best_score = score
                best_matrix = matrix
                best_pred = val_pred

    return best_matrix, best_pred


def calibrate_expert(
    expert,
    X_train_fit,
    y_train_fit,
    X_val,
    y_val,
    targets=None,
    off_diagonal_variants=None,
    clip_range=(0.45, 0.95),
    scales=(0.92, 0.96, 1.0, 1.04, 1.08),
    verbose=True,
):
    """
    Fit expert knowledge on train_fit and tune the confusion matrix on validation only.
    """
    if targets is None:
        targets = EXPERT_TARGET_CLASS_ACCURACY

    expert.fit(X_train_fit, y_train_fit)
    val_proba = expert.precompute_features(X_val)
    confusion_matrix, val_pred = _search_confusion_matrix(
        y_val, val_proba, targets,
        off_diagonal_variants=off_diagonal_variants,
        clip_range=clip_range,
        scales=scales,
    )

    calibrated = SimulatedExpert(
        confusion_matrix,
        random_state=RANDOM_STATE,
        domain_keywords=expert.domain_keywords,
        keyword_weight=expert.keyword_weight,
        name=expert.name,
        description=expert.description,
        sharpness=expert.sharpness,
        adaptive_keyword_weight=expert.adaptive_keyword_weight,
    )
    calibrated.pipeline = expert.pipeline
    val_pred = calibrated.predict_from_features(val_proba)
    pre_model = pre_model_accuracy(val_proba, y_val)

    if verbose:
        per_class = _per_class_accuracy(y_val, val_pred)
        print("[Task2] Competence diagonal:", np.round(np.diag(confusion_matrix), 3).tolist())
        print("[Task2] Validation expert accuracy:", round(float(accuracy_score(y_val, val_pred)), 4))
        print("[Task2] Pre-model (stage 1-2) standalone accuracy:", round(pre_model["overall"], 4))
        for label in range(NUM_CLASSES):
            print(
                f"  {CLASS_NAMES[label]}: target={targets[label]:.2f}, "
                f"observed={per_class[label]:.3f}, pre_model={pre_model['per_class'][label]:.3f}"
            )

    return {
        "expert": calibrated,
        "calibrated_reliability": np.diag(confusion_matrix).tolist(),
        "target_profile": targets,
        "validation_predictions": val_pred,
        "validation_per_class_accuracy": _per_class_accuracy(y_val, val_pred),
        "validation_overall_accuracy": float(accuracy_score(y_val, val_pred)),
        "pre_model_accuracy": pre_model,
        "error_distribution": {
            str(row): [
                (col, float(confusion_matrix[row, col]))
                for col in range(NUM_CLASSES)
                if col != row and confusion_matrix[row, col] > 0
            ]
            for row in range(NUM_CLASSES)
        },
        "design_rationale": (
            "Expert reads each article using a weaker MultinomialNB + keyword model, "
            "infers a latent topic from text, then samples a label from a class-conditional "
            "confusion matrix calibrated on validation. True labels are never used at inference."
        ),
    }
