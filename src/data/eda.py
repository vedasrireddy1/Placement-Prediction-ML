import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ingest import load_and_validate_data


def perform_eda():
    data_path = os.path.join("src", "data", "raw_placement_data.csv")

    df = load_and_validate_data(data_path)

    print("\n" + "=" * 50)
    print("--- 1. DATASET DIMENSIONS ---")
    print(f"Total Rows (Samples): {df.shape[0]}")
    print(f"Total Columns (Metrics): {df.shape[1]}")

    print("\n" + "=" * 50)
    print("--- 2. FEATURE NAMES & DATA TYPES ---")
    print(df.dtypes)

    print("\n" + "=" * 50)
    print("--- 3. MISSING VALUES & DUPLICATES ---")
    missing_vals = df.isnull().sum()

    if missing_vals.sum() > 0:
        print("Missing Values per Column:")
        print(missing_vals[missing_vals > 0])
    else:
        print("No missing values found.")

    duplicates = df.duplicated().sum()
    print(f"Duplicate Records Count: {duplicates}")

    print("\n" + "=" * 50)
    print("--- 4. SUMMARY STATISTICS ---")
    print(df.describe())

    print("\n" + "=" * 50)
    print("--- 5. CLASS IMBALANCE ANALYSIS ---")

    if "placement_status" in df.columns:
        class_counts = df["placement_status"].value_counts()
        class_percentages = df["placement_status"].value_counts(normalize=True) * 100

        print("Placement Status Counts:")
        print(class_counts)

        print("\nPlacement Status Percentages:")
        print(class_percentages)

    print("\n" + "=" * 50)
    print("--- 6. GENERATING VISUALIZATIONS ---")

    os.makedirs("reports/figures", exist_ok=True)

    sns.set_theme(style="whitegrid")

    numerical_df = df.select_dtypes(include=[np.number])

    plt.figure(figsize=(12, 9))
    corr_matrix = numerical_df.corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Matrix Heatmap")
    plt.tight_layout()
    plt.savefig("reports/figures/correlation_heatmap.png")
    plt.close()

    print("-> Saved correlation heatmap")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="cgpa",
        y="salary_package_lpa",
        hue="placement_status",
        alpha=0.7
    )
    plt.title("CGPA vs Salary Package")
    plt.tight_layout()
    plt.savefig("reports/figures/scatter_cgpa_salary.png")
    plt.close()

    print("-> Saved scatter plot")

    pairplot_cols = [
        "cgpa",
        "backlogs",
        "communication_skill_score",
        "internships_count",
        "salary_package_lpa"
    ]

    pp = sns.pairplot(
        df[pairplot_cols],
        diag_kind="kde",
        corner=True
    )
    pp.fig.suptitle(
        "Pairwise Relationships of Key Numerical Features",
        y=1.02
    )
    pp.savefig("reports/figures/pairplot_features.png")
    plt.close()

    print("-> Saved pair plot")

    plt.figure(figsize=(12, 8))
    sns.boxplot(data=numerical_df, orient="h")
    plt.title("Outlier Identification via Boxplots")
    plt.tight_layout()
    plt.savefig("reports/figures/outliers_boxplot.png")
    plt.close()

    print("-> Saved outlier boxplot")

    print("\nEDA Execution Complete.")
    print("Visualizations stored in reports/figures/")


if __name__ == "__main__":
    perform_eda()