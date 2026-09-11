import random


PAIRWISE_ROUNDS = 15
RANKING_ROUNDS = 2
RANKING_SET_SIZE = 10


def sample_pair(movies, seen_pairs):
    for _ in range(100):
        left, right = random.sample(movies, 2)
        key = tuple(sorted((left["id"], right["id"])))
        if key not in seen_pairs:
            seen_pairs.add(key)
            return left, right, seen_pairs
    left, right = random.sample(movies, 2)
    return left, right, seen_pairs


def sample_ranking_set(movies, seen_sets):
    for _ in range(100):
        selected = random.sample(movies, RANKING_SET_SIZE)
        key = tuple(sorted(movie["id"] for movie in selected))
        if key not in seen_sets:
            seen_sets.add(key)
            return selected, seen_sets
    selected = random.sample(movies, RANKING_SET_SIZE)
    return selected, seen_sets
