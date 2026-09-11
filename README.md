# Human-Centric AI — Course Projects

This repository contains four Django-based projects developed for the **Human-Centric
Artificial Intelligence** course. Each project is a self-contained Django app exploring a
different aspect of human-AI collaboration, accessible from a single shared home page.

**Author:** Saira Batool (670290)

## Projects

| # | Project | Summary |
|---|---------|---------|
| 1 | **Supervised Learning Interface** | Upload a CSV dataset, visualize it (scatter plots, correlation heatmap, class balance), and train a classification or regression model with configurable hyperparameters. |
| 2 | **Explainability** | Interpretability on the Palmer Penguins dataset: a regularization-controlled decision tree / logistic regression, hand-implemented PDP and ALE plots, and counterfactual explanations (MAD-weighted L1 distance). |
| 3 | **Active Learning for Learning-to-Defer** | Human-AI collaboration on AG News topic classification: a baseline classifier, two simulated experts, learning-to-defer strategies, and active-learning query strategies for expert competence discovery. Full results and reasoning are in the downloadable PDF report. |
| 4 | **Preference Elicitation** | A movie recommender user-study interface (pairwise comparisons vs. full ranking) built on a Bradley-Terry / Plackett-Luce preference model, with a consent flow and a downloadable study-design report. |

## Setup

```bash
git clone https://github.com/SairaBatool010//Saira_670290_HCAI_Project.git
cd HCAI_Project_Saira_Batool_2026

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

pip install -r requirements.txt

python manage.py migrate
python manage.py runserver
```

Then open **http://127.0.0.1:8000/** in a browser.

## Notes on Project 3

Project 3's dashboard (`/project3/`) reads from precomputed experiment artifacts
(models, metrics, and plots) that are already included in this repository under
`media/project3/experiments/`, so it works immediately without retraining anything.

If those artifacts were ever missing, they can be regenerated with:

```bash
python manage.py run_project3_task1
python manage.py run_project3_task2
python manage.py run_project3_phases5_10   # Tasks 3-4 + PDF report (Expert A)
python manage.py run_project3_expertB      # Tasks 2-4 + PDF report (Expert B)
```

## Project structure

```
pbl/            # Django project settings/urls
home/           # Landing page linking to all four projects
project1/       # Supervised Learning Interface
project2/       # Explainability
project3/       # Active Learning for Learning-to-Defer
project4/       # Preference Elicitation
static/         # Shared and per-project CSS
templates/      # Shared base template
media/          # Generated plots, reports, uploads, and Project 3's precomputed experiment artifacts
```
