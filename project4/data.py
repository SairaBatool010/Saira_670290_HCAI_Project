import os
import urllib.request

import numpy as np
import pandas as pd
from django.conf import settings
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.join(settings.BASE_DIR, "project4", "data")
DATA_PATH = os.path.join(DATA_DIR, "movie_metadata.csv")
DATA_URL = "https://raw.githubusercontent.com/sundeepblue/movie_rating_prediction/master/movie_metadata.csv"

TOP_GENRES = [
    "Action",
    "Adventure",
    "Comedy",
    "Drama",
    "Romance",
    "Thriller",
    "Crime",
    "Sci-Fi",
]

FEATURE_DESCRIPTIONS = [
    "IMDB score (quality signal)",
    "Duration in minutes",
    "Log budget",
    "Log gross revenue",
    "Log number of votes",
    "Release year (normalized)",
    "Movie Facebook likes (log)",
    "Genre indicators for major categories",
]


def _download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(DATA_PATH):
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)


def _parse_genres(value):
    if not isinstance(value, str):
        return []
    return [genre.strip() for genre in value.split("|") if genre.strip()]


def _safe_log(series):
    return np.log1p(series.clip(lower=0))


def build_feature_matrix(df):
    features = pd.DataFrame(index=df.index)
    features["imdb_score"] = pd.to_numeric(df["imdb_score"], errors="coerce")
    features["duration"] = pd.to_numeric(df["duration"], errors="coerce")
    features["log_budget"] = _safe_log(pd.to_numeric(df["budget"], errors="coerce"))
    features["log_gross"] = _safe_log(pd.to_numeric(df["gross"], errors="coerce"))
    features["log_votes"] = _safe_log(pd.to_numeric(df["num_voted_users"], errors="coerce"))
    features["title_year"] = pd.to_numeric(df["title_year"], errors="coerce")
    features["log_movie_likes"] = _safe_log(
        pd.to_numeric(df["movie_facebook_likes"], errors="coerce")
    )

    genres = df["genres"].apply(_parse_genres)
    for genre in TOP_GENRES:
        features[f"genre_{genre.lower()}"] = genres.apply(lambda items: int(genre in items))

    features = features.fillna(features.median(numeric_only=True))
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    feature_names = list(features.columns)
    return scaled, feature_names, scaler


def load_movies():
    _download_dataset()
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["movie_title", "imdb_score"]).copy()
    df["movie_title"] = df["movie_title"].astype(str).str.strip()
    df["director_name"] = df["director_name"].fillna("Unknown").astype(str)
    df["genres"] = df["genres"].fillna("").astype(str)
    df["plot_keywords"] = df["plot_keywords"].fillna("").astype(str)
    df = df.reset_index(drop=True)

    feature_matrix, feature_names, scaler = build_feature_matrix(df)
    movies = []
    for index, row in df.iterrows():
        movies.append(
            {
                "id": int(index),
                "title": row["movie_title"],
                "director": row["director_name"],
                "genres": row["genres"].replace("|", ", "),
                "year": int(row["title_year"]) if pd.notna(row["title_year"]) else "Unknown",
                "imdb_score": float(row["imdb_score"]),
                "features": feature_matrix[index],
            }
        )

    return {
        "movies": movies,
        "movies_by_id": {movie["id"]: movie for movie in movies},
        "feature_names": feature_names,
        "feature_matrix": feature_matrix,
        "scaler": scaler,
        "feature_descriptions": FEATURE_DESCRIPTIONS,
        "top_genres": TOP_GENRES,
    }
