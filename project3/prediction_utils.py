import numpy as np


def extract_prediction_features(probabilities):
    confidence = probabilities.max(axis=1)
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-12), axis=1)
    sorted_probs = np.sort(probabilities, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    predicted_class = probabilities.argmax(axis=1)
    return {
        "confidence": confidence,
        "entropy": entropy,
        "margin": margin,
        "predicted_class": predicted_class,
        "probabilities": probabilities,
    }


def competence_feature_column(predicted_class, competence_by_class, default=0.5):
    """Look up an (estimated) expert competence value per predicted class.

    This turns a per-class competence profile (known exactly in Task 3, where full
    expert labels are available, or only estimated from queried points in Task 4)
    into a per-example feature: "how good is the expert typically on articles the
    AI thinks belong to this class?" Confidence/entropy/margin alone only describe
    the AI's own uncertainty, not whether *this specific expert* is likely to be
    right when the AI is wrong -- this feature gives the deferral model that signal.
    """
    return np.array(
        [float(competence_by_class.get(int(label), default)) for label in predicted_class]
    )


def expected_competence_column(probabilities, competence_by_class, default=0.5):
    """Expected expert competence under the AI's full predicted probability
    distribution, not just its argmax class.

    Using only the argmax class misses a real, common case: when the AI is *wrong*,
    its argmax is -- by definition -- not the true class, so an article that is truly
    Business but which the AI (barely) misclassifies as Sci/Tech contributes nothing
    to a Business-indexed competence lookup, even though the AI's own probability
    vector likely still placed meaningful mass on Business for exactly that reason
    (Business/Sci-Tech is a well-known confusion pair). Weighting the competence
    lookup by the AI's full probability vector lets that residual mass contribute
    proportionally, rather than being thrown away by a hard argmax.
    """
    competence_vector = np.array(
        [float(competence_by_class.get(c, default)) for c in range(probabilities.shape[1])]
    )
    return probabilities @ competence_vector


def build_deferral_feature_matrix(features, competence_by_class):
    competence = competence_feature_column(features["predicted_class"], competence_by_class)
    expected_competence = expected_competence_column(features["probabilities"], competence_by_class)
    uncertainty = 1.0 - features["confidence"]
    return np.column_stack(
        [
            features["confidence"],
            features["entropy"],
            features["margin"],
            features["predicted_class"],
            uncertainty,
            competence,
            # Explicit interaction: AI uncertainty alone, or expert competence alone,
            # is not enough -- deferring only pays off when the AI is likely wrong
            # AND the expert is likely right for that predicted class at the same
            # time. This term gives the (linear) model direct access to that joint
            # condition instead of having to approximate it from the two factors
            # separately, which is what let Expert B's deferral policy find and
            # exploit real value instead of collapsing to "never defer" again.
            uncertainty * competence,
            expected_competence,
            uncertainty * expected_competence,
        ]
    )
