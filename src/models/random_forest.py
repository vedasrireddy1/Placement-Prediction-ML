from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "src" / "data" / "raw_placement_data.csv"
OUT = ROOT / "reports" / "figures"

OUT.mkdir(parents=True, exist_ok=True)


FEATURES = [
    "branch",
    "college_tier",
    "cgpa",
    "backlogs",
    "coding_skill_score",
    "communication_skill_score",
    "internships_count",
    "projects_count"
]

TARGET = "placement_status"


def load():
    df = pd.read_csv(DATA)

    df.columns = df.columns.str.strip()

    df = df[FEATURES + [TARGET]].dropna()

    if not pd.api.types.is_numeric_dtype(df["college_tier"]):
        df["college_tier"] = (
            df["college_tier"]
            .astype(str)
            .str.extract(r"(\d+)")
            .astype(float)
        )

    X = pd.get_dummies(
        df[FEATURES],
        columns=["branch"],
        dtype=float
    )

    y = df[TARGET]

    if not pd.api.types.is_numeric_dtype(y):
        labels = sorted(y.astype(str).unique())

        if len(labels) != 2:
            raise ValueError(
                "placement_status must be binary"
            )

        y = y.astype(str).map({
            labels[0]: 0,
            labels[1]: 1
        })

    return X, y


def score(model, X, y):
    pred = model.predict(X)

    return [
        round(accuracy_score(y, pred), 4),
        round(
            precision_score(
                y,
                pred,
                zero_division=0
            ),
            4
        ),
        round(
            recall_score(
                y,
                pred,
                zero_division=0
            ),
            4
        ),
        round(
            f1_score(
                y,
                pred,
                zero_division=0
            ),
            4
        )
    ]


def run():

    X, y = load()

    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    print("\n" + "=" * 50)
    print("--- DECISION TREE VS RANDOM FOREST ---")

    dt = DecisionTreeClassifier(
        max_depth=12,
        random_state=42
    )

    dt.fit(Xtr, ytr)

    dt_score = score(
        dt,
        Xte,
        yte
    )

    print(
        "Decision Tree "
        "[accuracy, precision, recall, F1]:",
        dt_score
    )


    rf = RandomForestClassifier(
        n_estimators=100,
        max_features="sqrt",
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(Xtr, ytr)

    rf_score = score(
        rf,
        Xte,
        yte
    )

    print(
        "Random Forest "
        "[accuracy, precision, recall, F1]:",
        rf_score
    )

    print(
        f"OOB score: {rf.oob_score_:.4f}"
    )

    print(
        f"OOB error: "
        f"{1 - rf.oob_score_:.4f}"
    )


    print("\n" + "=" * 50)
    print("--- EFFECT OF NUMBER OF TREES ---")

    counts = [
        10,
        25,
        50,
        100,
        200
    ]

    oob_errors = []
    test_accuracies = []

    for n in counts:

        model = RandomForestClassifier(
            n_estimators=n,
            max_features="sqrt",
            oob_score=True,
            random_state=42,
            n_jobs=-1
        )

        model.fit(Xtr, ytr)

        oob_error = (
            1 - model.oob_score_
        )

        test_accuracy = accuracy_score(
            yte,
            model.predict(Xte)
        )

        oob_errors.append(
            oob_error
        )

        test_accuracies.append(
            test_accuracy
        )

        print(
            f"Trees: {n} | "
            f"OOB Error: {oob_error:.4f} | "
            f"Test Accuracy: {test_accuracy:.4f}"
        )


    plt.figure(figsize=(9, 6))

    plt.plot(
        counts,
        oob_errors,
        marker="o",
        label="OOB Error"
    )

    plt.plot(
        counts,
        test_accuracies,
        marker="o",
        label="Test Accuracy"
    )

    plt.xlabel("Number of Trees")
    plt.ylabel("Value")

    plt.title(
        "Effect of Number of Trees - Placement Prediction"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUT / "random_forest_number_of_trees.png",
        dpi=150
    )

    plt.close()

    print(
        "Saved: "
        "reports/figures/random_forest_number_of_trees.png"
    )


    print("\n" + "=" * 50)
    print("--- FEATURE SUBSAMPLING ---")

    options = [
        "sqrt",
        "log2",
        None
    ]

    labels = [
        "sqrt",
        "log2",
        "all features"
    ]

    values = []

    for option in options:

        model = RandomForestClassifier(
            n_estimators=100,
            max_features=option,
            random_state=42,
            n_jobs=-1
        )

        model.fit(Xtr, ytr)

        accuracy = accuracy_score(
            yte,
            model.predict(Xte)
        )

        values.append(
            accuracy
        )

    for label, value in zip(
        labels,
        values
    ):
        print(
            f"{label}: {value:.4f}"
        )


    plt.figure(figsize=(8, 6))

    plt.bar(
        labels,
        values
    )

    plt.ylabel("Test Accuracy")

    plt.title(
        "Feature Subsampling - Placement Prediction"
    )

    plt.ylim(0, 1)

    for i, value in enumerate(values):

        plt.text(
            i,
            value + 0.02,
            f"{value:.4f}",
            ha="center"
        )

    plt.tight_layout()

    plt.savefig(
        OUT / "random_forest_feature_subsampling.png",
        dpi=150
    )

    plt.close()

    print(
        "Saved: "
        "reports/figures/random_forest_feature_subsampling.png"
    )


    print("\n" + "=" * 50)
    print(
        "Week 9 Random Forest execution complete."
    )


if __name__ == "__main__":
    run()