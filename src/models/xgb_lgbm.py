from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import shap


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

    if not pd.api.types.is_numeric_dtype(
        df["college_tier"]
    ):
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

        labels = sorted(
            y.astype(str).unique()
        )

        if len(labels) != 2:
            raise ValueError(
                "placement_status must be binary"
            )

        y = y.astype(str).map({
            labels[0]: 0,
            labels[1]: 1
        })

    return X, y


def shap_report(model, X, name):

    print(
        f"\nGenerating SHAP analysis for {name}..."
    )

    sample_size = min(
        len(X),
        2000
    )

    X_sample = X.sample(
        sample_size,
        random_state=42
    )

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(
        X_sample
    )

    if isinstance(shap_values, list):

        values = (
            shap_values[1]
            if len(shap_values) > 1
            else shap_values[0]
        )

    elif len(
        getattr(shap_values, "shape", ())
    ) == 3:

        values = shap_values[:, :, 1]

    else:

        values = shap_values

    importance = pd.DataFrame({
        "Feature": X_sample.columns,
        "Mean_Absolute_SHAP": np.abs(
            values
        ).mean(axis=0)
    })

    importance = importance.sort_values(
        "Mean_Absolute_SHAP",
        ascending=False
    )

    importance.to_csv(
        OUT / f"{name}_shap_feature_importance.csv",
        index=False
    )

    print(
        f"\nTop features ({name}):"
    )

    print(
        importance.head(10).to_string(
            index=False
        )
    )

    plt.figure()

    shap.summary_plot(
        values,
        X_sample,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        OUT / f"{name}_shap_summary.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    top_feature = importance.iloc[0]["Feature"]

    plt.figure()

    shap.dependence_plot(
        top_feature,
        values,
        X_sample,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        OUT / f"{name}_shap_dependence.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"SHAP analysis saved for {name}."
    )


def run():

    print("\nLoading placement dataset...")

    X, y = load()

    print(
        f"Samples: {X.shape[0]}"
    )

    print(
        f"Features after encoding: "
        f"{X.shape[1]}"
    )

    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    models = {

        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        ),

        "lightgbm": LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbosity=-1,
            n_jobs=-1
        )
    }

    results = []

    for name, model in models.items():

        print("\n" + "=" * 50)
        print(
            f"--- {name.upper()} ---"
        )

        start = time.perf_counter()

        model.fit(
            Xtr,
            ytr
        )

        training_time = (
            time.perf_counter()
            - start
        )

        pred = model.predict(Xte)

        accuracy = accuracy_score(
            yte,
            pred
        )

        print(
            f"Accuracy: {accuracy:.4f}"
        )

        print(
            f"Training Time: "
            f"{training_time:.4f} seconds"
        )

        print("\nClassification Report:")

        print(
            classification_report(
                yte,
                pred
            )
        )

        results.append([
            name,
            accuracy,
            training_time
        ])

        shap_report(
            model,
            Xte,
            name
        )

    results_df = pd.DataFrame(
        results,
        columns=[
            "Model",
            "Accuracy",
            "Training_Time"
        ]
    )

    results_df.to_csv(
        OUT / "xgb_lgbm_comparison.csv",
        index=False
    )

    print("\n" + "=" * 50)
    print("--- MODEL COMPARISON ---")

    print(
        results_df.to_string(
            index=False
        )
    )

    plt.figure(figsize=(8, 6))

    plt.bar(
        results_df["Model"],
        results_df["Accuracy"]
    )

    plt.ylabel("Accuracy")

    plt.title(
        "XGBoost vs LightGBM Accuracy"
    )

    plt.ylim(0, 1)

    for i, value in enumerate(
        results_df["Accuracy"]
    ):

        plt.text(
            i,
            value + 0.02,
            f"{value:.4f}",
            ha="center"
        )

    plt.tight_layout()

    plt.savefig(
        OUT / "xgb_lgbm_accuracy_comparison.png",
        dpi=150
    )

    plt.close()

    print(
        "\nSaved: "
        "reports/figures/xgb_lgbm_accuracy_comparison.png"
    )

    print(
        "\nWeek 10 XGBoost + LightGBM + SHAP execution complete."
    )


if __name__ == "__main__":
    run()