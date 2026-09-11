import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


def utility_weights(feature_matrix, weights):
    return feature_matrix @ weights


def pairwise_log_likelihood(weights, feature_matrix, winner_ids, loser_ids):
    utilities = utility_weights(feature_matrix, weights)
    delta = utilities[winner_ids] - utilities[loser_ids]
    return float(np.sum(-np.logaddexp(0, -delta)))


def ranking_log_likelihood(weights, feature_matrix, ranking_ids):
    utilities = utility_weights(feature_matrix, weights)
    total = 0.0
    for ranking in ranking_ids:
        remaining = list(ranking)
        while remaining:
            current = remaining.pop(0)
            denominator = logsumexp(utilities[remaining + [current]])
            total += utilities[current] - denominator
    return float(total)


def fit_preference_weights(
    feature_matrix, pairwise_observations=None, ranking_observations=None, l2_penalty=1.0
):
    """Maximum a posteriori estimate of the preference vector w.

    With as few as 15 comparisons and ~15 features, the plain Bradley-Terry / Plackett-Luce
    MLE is only weakly identified: if the observed choices are (close to) perfectly
    separable along some direction, the unregularized MLE diverges to infinite weight
    magnitude (a well-known degeneracy of logistic-type likelihoods on separable data).
    We add a Gaussian prior on w (equivalently, an L2 / ridge penalty on the objective),
    which keeps the estimate finite and well-behaved even for a handful of responses.
    """
    num_features = feature_matrix.shape[1]
    weights0 = np.zeros(num_features)

    def objective(weights):
        value = 0.0
        if pairwise_observations:
            winners, losers = zip(*pairwise_observations)
            value += pairwise_log_likelihood(weights, feature_matrix, np.array(winners), np.array(losers))
        if ranking_observations:
            value += ranking_log_likelihood(weights, feature_matrix, ranking_observations)
        value -= 0.5 * l2_penalty * float(np.sum(weights ** 2))
        return -value

    result = minimize(objective, weights0, method="L-BFGS-B")
    return result.x


def top_recommendations(movies, weights, top_k=5):
    scores = []
    for movie in movies:
        score = float(movie["features"] @ weights)
        scores.append((score, movie))
    scores.sort(key=lambda item: item[0], reverse=True)
    return [
        {
            "title": movie["title"],
            "genres": movie["genres"],
            "year": movie["year"],
            "imdb_score": movie["imdb_score"],
            "score": round(score, 4),
        }
        for score, movie in scores[:top_k]
    ]


def bradley_terry_pairwise_probability(weights, feature_matrix, left_id, right_id):
    utilities = utility_weights(feature_matrix, weights)
    delta = utilities[left_id] - utilities[right_id]
    return float(1.0 / (1.0 + np.exp(-delta)))


def plackett_luce_ranking_probability(weights, feature_matrix, ranking_ids):
    utilities = utility_weights(feature_matrix, weights)
    log_prob = 0.0
    remaining = list(ranking_ids)
    while remaining:
        current = remaining.pop(0)
        denominator = logsumexp(utilities[remaining + [current]])
        log_prob += utilities[current] - denominator
    return float(np.exp(log_prob))
