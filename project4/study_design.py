STUDY_DESIGN = {
    "title": "Comparing Pairwise and Ranking Preference Elicitation for Movie Recommendation",
    "hypothesis": (
        "H1: The ranking interface (Design 2) yields more accurate estimates of the latent "
        "preference vector w than the pairwise interface (Design 1), at the cost of longer "
        "interaction time and higher cognitive load."
    ),
    "design_type": "Between-subjects randomized controlled study",
    "participants": {
        "target_n": 40,
        "inclusion": "Adults who watch movies at least occasionally and can read English.",
        "recruitment": "University mailing lists, social media, or Prolific with a fair compensation.",
    },
    "conditions": {
        "design1": f"{15} pairwise comparisons of two movies each",
        "design2": f"{2} ranking tasks with {10} movies each",
    },
    "procedure": [
        "Obtain informed consent and collect demographic information.",
        "Randomly assign participant to Design 1 or Design 2.",
        "Show standardized instructions with one practice trial.",
        "Run the elicitation tasks with movies sampled uniformly at random.",
        "Collect task completion time and optional NASA-TLX workload ratings.",
        "Estimate w with the Bradley-Terry / Plackett-Luce model.",
        "Evaluate on holdout pairwise comparisons not shown during elicitation.",
        "Debrief participants and provide optional recommendation preview.",
    ],
    "confounds": [
        "The IMDB score is deliberately hidden from participants during the elicitation "
        "tasks, even though it is used as a model feature. Showing a crowd rating while "
        "asking for personal preference risks anchoring participants on the aggregate "
        "opinion of others rather than eliciting their own taste.",
        "Movies are sampled uniformly at random rather than adaptively; this keeps the "
        "comparison between designs fair but is not the most sample-efficient elicitation "
        "strategy (see the note on adaptive selection in the task description).",
        "Order effects (fatigue, learning) are not counterbalanced across rounds in this "
        "demonstration interface; a full deployment should randomize or counterbalance "
        "round order.",
    ],
    "metrics": [
        "Holdout pairwise prediction accuracy (Bradley-Terry log-likelihood on unseen pairs).",
        "Top-k recommendation overlap with participant validation choices.",
        "Total study duration and time per task.",
        "Subjective workload and preference confidence (post-study questionnaire).",
    ],
    "analysis_plan": [
        "Compare holdout accuracy between Design 1 and Design 2 using a two-sample t-test.",
        "Compare completion time with Mann-Whitney U test.",
        "Report effect sizes and confidence intervals.",
        "Perform sanity checks for random responding and incomplete sessions.",
    ],
}

RANKING_MODEL_TEXT = (
    "The Bradley-Terry model is extended to full rankings using the Plackett-Luce model. "
    "For a ranking i<sub>1</sub> &gt; i<sub>2</sub> &gt; ... &gt; i<sub>n</sub> of n items (read "
    "\"&gt;\" as \"preferred to\"), the probability of observing the entire order is decomposed as "
    "a sequence of independent choices, each picking the most preferred item among those not yet "
    "ranked:<br/><br/>"
    "P(ranking) = [product over k = 1 .. n of] exp(u<sub>ik</sub>) divided by the sum of "
    "exp(u<sub>ij</sub>) over all items i<sub>j</sub> still remaining at step k,<br/><br/>"
    "where u<sub>i</sub> = w<sup>T</sup>x<sub>i</sub> is the latent utility of item i. Each factor "
    "in the product is itself a standard Bradley-Terry choice probability among the items not yet "
    "ranked, so pairwise comparison (Design 1) is exactly the special case n = 2 (a single choice "
    "between two remaining items). This formulation lets both interfaces estimate the same latent "
    "vector w with the same likelihood family, which is what makes a direct comparison between "
    "them meaningful."
)

FEATURE_JUSTIFICATION = (
    "Movie features combine a quality signal (IMDB score), scale (log budget, log gross "
    "revenue, log vote count), temporal context (release year), popularity (log Facebook "
    "likes), and genre indicators for the eight most common genres. These attributes are "
    "interpretable, available for nearly all movies in the dataset, and align with factors "
    "users commonly cite when explaining their movie preferences. Monetary and count "
    "variables are log-transformed to reduce the influence of extreme outliers (e.g. "
    "blockbuster budgets), and all features are standardized to zero mean and unit variance "
    "so that the fitted weight vector w is comparable across features."
)

MODELING_NOTES = (
    "The preference vector w is estimated by maximizing the Bradley-Terry / Plackett-Luce "
    "log-likelihood of the observed choices, plus an L2 (ridge) penalty on w. With as few as "
    "15 pairwise comparisons and roughly 15 features, the unregularized maximum-likelihood "
    "estimate is only weakly identified: if a participant's choices happen to be perfectly "
    "separable along some direction in feature space, the unregularized estimate diverges to "
    "infinite magnitude, a well-known degeneracy of logistic-type likelihoods on separable "
    "data. The L2 penalty corresponds to a Gaussian prior on w and keeps the estimate finite "
    "and numerically stable even after a small number of responses."
)
