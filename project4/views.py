import os
import random
import time

from django.http import FileResponse, Http404
from django.shortcuts import redirect, render

from .data import load_movies
from .forms import ConsentForm, StartStudyForm
from .preference_model import fit_preference_weights, top_recommendations
from .report import generate_report_pdf
from .selection import PAIRWISE_ROUNDS, RANKING_ROUNDS, RANKING_SET_SIZE, sample_pair, sample_ranking_set
from .study_design import STUDY_DESIGN


_MOVIE_DATA = None
_REPORT = None


def _get_movie_data():
    global _MOVIE_DATA
    if _MOVIE_DATA is None:
        _MOVIE_DATA = load_movies()
    return _MOVIE_DATA


def _get_report():
    global _REPORT
    if _REPORT is None:
        _REPORT = generate_report_pdf()
    return _REPORT


def _init_session(request, participant_id, design, demographics):
    request.session["study"] = {
        "participant_id": participant_id,
        "design": design,
        "demographics": demographics,
        "round_index": 0,
        "pairwise_observations": [],
        "ranking_observations": [],
        "seen_pairs": [],
        "seen_sets": [],
        "current_pair": None,
        "current_ranking_set": None,
        "started_at": time.time(),
    }


def _get_study(request):
    return request.session.get("study")


def index(request):
    report_url, _ = _get_report()
    movie_data = _get_movie_data()
    context = {
        "report_url": report_url,
        "movie_count": len(movie_data["movies"]),
        "study_design": STUDY_DESIGN,
        "start_form": StartStudyForm(),
    }
    return render(request, "project4/index.html", context)


def download_report(request):
    _, report_path = _get_report()
    if not os.path.exists(report_path):
        raise Http404("Report not found.")
    return FileResponse(open(report_path, "rb"), as_attachment=True, filename="project4_report.pdf")


def start_study(request):
    if request.method != "POST":
        return redirect("project4:index")

    form = StartStudyForm(request.POST)
    if not form.is_valid():
        report_url, _ = _get_report()
        return render(
            request,
            "project4/index.html",
            {
                "report_url": report_url,
                "movie_count": len(_get_movie_data()["movies"]),
                "study_design": STUDY_DESIGN,
                "start_form": form,
                "error": "Please provide a valid participant ID.",
            },
        )

    design = form.cleaned_data["design"]
    if design == "random":
        design = random.choice(["pairwise", "ranking"])

    request.session["pending_study"] = {
        "participant_id": form.cleaned_data["participant_id"],
        "design": design,
    }
    return redirect("project4:consent")


def consent(request):
    pending = request.session.get("pending_study")
    if not pending:
        return redirect("project4:index")

    if request.method == "POST":
        form = ConsentForm(request.POST)
        if form.is_valid():
            demographics = {
                "age_range": form.cleaned_data["age_range"],
                "movie_frequency": form.cleaned_data["movie_frequency"],
            }
            _init_session(request, pending["participant_id"], pending["design"], demographics)
            del request.session["pending_study"]
            return redirect("project4:instructions")
    else:
        form = ConsentForm()

    return render(request, "project4/consent.html", {"form": form})


def instructions(request):
    study = _get_study(request)
    if not study:
        return redirect("project4:index")

    total_rounds = PAIRWISE_ROUNDS if study["design"] == "pairwise" else RANKING_ROUNDS
    context = {
        "design": study["design"],
        "total_rounds": total_rounds,
        "ranking_set_size": RANKING_SET_SIZE,
    }
    return render(request, "project4/instructions.html", context)


def task(request):
    study = _get_study(request)
    if not study:
        return redirect("project4:index")

    movie_data = _get_movie_data()
    movies = movie_data["movies"]

    if study["design"] == "pairwise":
        return _pairwise_task(request, study, movies, movie_data)
    return _ranking_task(request, study, movies, movie_data)


def _pairwise_task(request, study, movies, movie_data):
    error = None

    if request.method == "POST":
        winner_id = request.POST.get("winner_id")
        loser_id = request.POST.get("loser_id")
        current_pair = study.get("current_pair")

        valid = (
            winner_id is not None
            and loser_id is not None
            and winner_id.isdigit()
            and loser_id.isdigit()
            and current_pair is not None
            and {int(winner_id), int(loser_id)} == set(current_pair)
        )

        if valid:
            study["pairwise_observations"].append([int(winner_id), int(loser_id)])
            study["round_index"] += 1
            study["current_pair"] = None
            request.session["study"] = study
            request.session.modified = True

            if study["round_index"] >= PAIRWISE_ROUNDS:
                return redirect("project4:complete")
        else:
            error = "Your response could not be recorded. Please choose one of the two movies shown below."

    if study["round_index"] >= PAIRWISE_ROUNDS:
        return redirect("project4:complete")

    if study.get("current_pair") and error:
        movies_by_id = movie_data["movies_by_id"]
        left = movies_by_id[study["current_pair"][0]]
        right = movies_by_id[study["current_pair"][1]]
    else:
        seen_pairs = {tuple(pair) for pair in study["seen_pairs"]}
        left, right, seen_pairs = sample_pair(movies, seen_pairs)
        study["seen_pairs"] = [list(item) for item in seen_pairs]
        study["current_pair"] = [left["id"], right["id"]]
        request.session["study"] = study
        request.session.modified = True

    context = {
        "design": "pairwise",
        "round_number": study["round_index"] + 1,
        "total_rounds": PAIRWISE_ROUNDS,
        "left_movie": left,
        "right_movie": right,
        "error": error,
    }
    return render(request, "project4/task_pairwise.html", context)


def _parse_ranking_submission(post_data, expected_movie_ids):
    """Validate a ranking submission. Returns (ranking_ids, error_message)."""
    rank_by_movie = {}
    for key, value in post_data.items():
        if not key.startswith("rank_"):
            continue
        movie_id_part = key.replace("rank_", "", 1)
        if not movie_id_part.isdigit():
            continue
        movie_id = int(movie_id_part)
        if not value.isdigit():
            return None, "Every movie must be assigned a rank."
        rank_by_movie[movie_id] = int(value)

    submitted_ids = set(rank_by_movie.keys())
    if submitted_ids != set(expected_movie_ids):
        return None, "The submitted ranking does not match the movies shown. Please try again."

    ranks_used = list(rank_by_movie.values())
    if sorted(ranks_used) != list(range(1, len(expected_movie_ids) + 1)):
        return None, "Each rank from 1 to {} must be used exactly once.".format(len(expected_movie_ids))

    ranking_ids = sorted(rank_by_movie.keys(), key=lambda movie_id: rank_by_movie[movie_id])
    return ranking_ids, None


def _ranking_task(request, study, movies, movie_data):
    error = None

    if request.method == "POST" and study.get("current_ranking_set"):
        ranking_ids, error = _parse_ranking_submission(request.POST, study["current_ranking_set"])
        if ranking_ids is not None:
            study["ranking_observations"].append(ranking_ids)
            study["round_index"] += 1
            study["current_ranking_set"] = None
            request.session["study"] = study
            request.session.modified = True

            if study["round_index"] >= RANKING_ROUNDS:
                return redirect("project4:complete")

    if study["round_index"] >= RANKING_ROUNDS:
        return redirect("project4:complete")

    movies_by_id = movie_data["movies_by_id"]
    if study.get("current_ranking_set") and error:
        selected = [movies_by_id[movie_id] for movie_id in study["current_ranking_set"]]
    else:
        seen_sets = {tuple(item) for item in study["seen_sets"]}
        selected, seen_sets = sample_ranking_set(movies, seen_sets)
        study["seen_sets"] = [list(item) for item in seen_sets]
        study["current_ranking_set"] = [movie["id"] for movie in selected]
        request.session["study"] = study
        request.session.modified = True

    context = {
        "design": "ranking",
        "round_number": study["round_index"] + 1,
        "total_rounds": RANKING_ROUNDS,
        "movies": selected,
        "ranking_set_size": RANKING_SET_SIZE,
        "rank_options": list(range(1, RANKING_SET_SIZE + 1)),
        "error": error,
    }
    return render(request, "project4/task_ranking.html", context)


def complete(request):
    study = _get_study(request)
    if not study:
        return redirect("project4:index")

    movie_data = _get_movie_data()
    feature_matrix = movie_data["feature_matrix"]
    pairwise = [tuple(item) for item in study["pairwise_observations"]]
    rankings = study["ranking_observations"]

    weights = fit_preference_weights(
        feature_matrix,
        pairwise_observations=pairwise if pairwise else None,
        ranking_observations=rankings if rankings else None,
    )
    recommendations = top_recommendations(movie_data["movies"], weights, top_k=5)
    elapsed = round(time.time() - study.get("started_at", time.time()), 1)

    context = {
        "participant_id": study["participant_id"],
        "design": study["design"],
        "pairwise_count": len(pairwise),
        "ranking_count": len(rankings),
        "recommendations": recommendations,
        "elapsed_seconds": elapsed,
    }
    return render(request, "project4/complete.html", context)
