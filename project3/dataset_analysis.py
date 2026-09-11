import json
import os

import numpy as np

from .config import TASK1_DIR


def _class_distribution(labels, class_names):
    counts = {class_names[i]: int(np.sum(labels == i)) for i in range(len(class_names))}
    total = len(labels)
    percentages = {name: round(count / total * 100, 2) for name, count in counts.items()}
    return {"counts": counts, "percentages": percentages, "total": total}


def _text_length_stats(texts):
    lengths = np.array([len(text.split()) for text in texts], dtype=int)
    return {
        "average_word_count": round(float(lengths.mean()), 2),
        "min_word_count": int(lengths.min()),
        "max_word_count": int(lengths.max()),
        "median_word_count": float(np.median(lengths)),
    }


def _representative_examples(texts, labels, class_names, n_examples=2, random_state=42):
    rng = np.random.default_rng(random_state)
    examples = {}
    for class_id, class_name in enumerate(class_names):
        indices = np.where(labels == class_id)[0]
        chosen = rng.choice(indices, size=min(n_examples, len(indices)), replace=False)
        examples[class_name] = [
            {"index": int(i), "text_preview": texts[int(i)][:240]} for i in chosen
        ]
    return examples


def analyze_dataset(splits):
    class_names = splits["class_names"]
    train_dist = _class_distribution(splits["y_train_full"], class_names)
    val_dist = _class_distribution(splits["y_val"], class_names)
    test_dist = _class_distribution(splits["y_test"], class_names)

    train_lengths = _text_length_stats(splits["X_train_full"])
    test_lengths = _text_length_stats(splits["X_test"])

    train_pct_values = list(train_dist["percentages"].values())
    is_balanced = max(train_pct_values) - min(train_pct_values) < 5.0

    report = {
        "dataset_name": "AG News",
        "random_state": splits["random_state"],
        "val_size": splits["val_size"],
        "splits": {
            "train_full": len(splits["X_train_full"]),
            "train_fit": len(splits["X_train_fit"]),
            "validation": len(splits["X_val"]),
            "test_official": len(splits["X_test"]),
        },
        "num_classes": len(class_names),
        "class_names": class_names,
        "class_distribution": {
            "train_full": train_dist,
            "validation": val_dist,
            "test_official": test_dist,
        },
        "text_length_words": {
            "train_full": train_lengths,
            "test_official": test_lengths,
        },
        "representative_examples": _representative_examples(
            splits["X_train_full"],
            splits["y_train_full"],
            class_names,
            random_state=splits["random_state"],
        ),
        "is_balanced": is_balanced,
        "balance_note": (
            "Classes are approximately balanced in AG News."
            if is_balanced
            else "Classes show noticeable imbalance; use stratified splits and macro metrics."
        ),
    }
    return report


def save_dataset_report(report, output_dir=TASK1_DIR):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "phase1_dataset_report.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return path
