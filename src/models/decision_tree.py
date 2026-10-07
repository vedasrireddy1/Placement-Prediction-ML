from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score


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
    print("--- DECISION TREE CLASSIFICATION ---")

    for name, criterion in [
        ("gini", "gini"),
        ("entropy", "entropy")
    ]:

        model = DecisionTreeClassifier(
            criterion=criterion,
            random_state=42,
            max_depth=12
        )

        model.fit(Xtr, ytr)

        pred = model.predict(Xte)

        accuracy = accuracy_score(
            yte,
            pred
        )

        print(
            f"{name.title()} test accuracy: "
            f"{accuracy:.4f}"
        )

        plt.figure(figsize=(18, 10))

        plot_tree(
            model,
            feature_names=X.columns.tolist(),
            class_names=[
                str(x) for x in sorted(y.unique())
            ],
            filled=True,
            max_depth=4,
            fontsize=7
        )

        plt.title(
            f"Placement Decision Tree - {name.title()}"
        )

        plt.tight_layout()

        plt.savefig(
            OUT / f"decision_tree_{name}.png",
            dpi=150
        )

        plt.close()

    print(
        "\nDecision tree visualizations saved."
    )


    print("\n" + "=" * 50)
    print("--- COST COMPLEXITY PRUNING ---")

    pruning_base = DecisionTreeClassifier(
        random_state=42,
        max_depth=12
    )

    path = pruning_base.cost_complexity_pruning_path(
        Xtr,
        ytr
    )

    alphas = path.ccp_alphas

    if len(alphas) > 30:

        idx = np.linspace(
            0,
            len(alphas) - 1,
            30,
            dtype=int
        )

        alphas = alphas[idx]

    rows = []

    for alpha in alphas:

        model = DecisionTreeClassifier(
            ccp_alpha=alpha,
            max_depth=12,
            random_state=42
        )

        model.fit(
            Xtr,
            ytr
        )

        train_pred = model.predict(Xtr)
        test_pred = model.predict(Xte)

        train_accuracy = accuracy_score(
            ytr,
            train_pred
        )

        test_accuracy = accuracy_score(
            yte,
            test_pred
        )

        rows.append([
            alpha,
            train_accuracy,
            test_accuracy,
            model.tree_.node_count
        ])

    pruning = pd.DataFrame(
        rows,
        columns=[
            "alpha",
            "train_accuracy",
            "test_accuracy",
            "nodes"
        ]
    )

    best = pruning.loc[
        pruning["test_accuracy"].idxmax()
    ]

    print(
        f"Best CCP alpha: "
        f"{best['alpha']:.8f}"
    )

    print(
        f"Pruned test accuracy: "
        f"{best['test_accuracy']:.4f}"
    )

    plt.figure(figsize=(9, 6))

    plt.plot(
        pruning["alpha"],
        pruning["train_accuracy"],
        marker="o",
        label="Training"
    )

    plt.plot(
        pruning["alpha"],
        pruning["test_accuracy"],
        marker="o",
        label="Testing"
    )

    plt.xlabel("CCP Alpha")
    plt.ylabel("Accuracy")

    plt.title(
        "Cost Complexity Pruning - Placement Prediction"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT / "decision_tree_ccp.png",
        dpi=150
    )

    plt.close()

    print(
        "Saved: reports/figures/decision_tree_ccp.png"
    )


    print("\n" + "=" * 50)
    print("--- EFFECT OF TREE DEPTH ---")

    depths = range(1, 16)

    train_scores = []
    test_scores = []

    for depth in depths:

        model = DecisionTreeClassifier(
            max_depth=depth,
            random_state=42
        )

        model.fit(
            Xtr,
            ytr
        )

        train_scores.append(
            accuracy_score(
                ytr,
                model.predict(Xtr)
            )
        )

        test_scores.append(
            accuracy_score(
                yte,
                model.predict(Xte)
            )
        )

    best_depth = list(depths)[
        int(np.argmax(test_scores))
    ]

    best_test_accuracy = max(
        test_scores
    )

    print(
        f"Best depth: {best_depth}"
    )

    print(
        f"Best test accuracy: "
        f"{best_test_accuracy:.4f}"
    )

    plt.figure(figsize=(9, 6))

    plt.plot(
        depths,
        train_scores,
        marker="o",
        label="Training"
    )

    plt.plot(
        depths,
        test_scores,
        marker="o",
        label="Testing"
    )

    plt.xlabel("Maximum Depth")
    plt.ylabel("Accuracy")

    plt.title(
        "Effect of Tree Depth - Placement Prediction"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT / "decision_tree_depth.png",
        dpi=150
    )

    plt.close()

    print(
        "Saved: reports/figures/decision_tree_depth.png"
    )

    print("\n" + "=" * 50)
    print(
        "Week 8 Decision Tree execution complete."
    )


if __name__ == "__main__":
    run()