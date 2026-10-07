import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data.ingest import load_and_validate_data


def train_linear_regression_ls():
    data_path = os.path.join("src", "data", "raw_placement_data.csv")

    df = load_and_validate_data(data_path)

    feature_cols = ["cgpa", "communication_skill_score"]
    target_col = "salary_package_lpa"

    df_clean = df.dropna(
        subset=feature_cols + [target_col]
    ).copy()

    X = df_clean[feature_cols].values
    y = df_clean[target_col].values.reshape(-1, 1)

    n = X.shape[0]

    X_design = np.hstack([
        np.ones((n, 1)),
        X
    ])

    xtx = np.dot(X_design.T, X_design)

    try:
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        xtx_inv = np.linalg.pinv(xtx)

    xty = np.dot(X_design.T, y)

    w = np.dot(xtx_inv, xty)

    y_pred = np.dot(X_design, w)

    error = 0.5 * np.sum((y_pred - y) ** 2)

    print("\n" + "=" * 50)
    print("--- LINEAR REGRESSION USING LEAST SQUARES ---")
    print(f"Samples: {n}")
    print(f"Features: {feature_cols}")

    print("\n--- Model Parameters ---")
    print(f"Intercept: {w[0, 0]:.4f}")
    print(f"CGPA coefficient: {w[1, 0]:.4f}")
    print(
        f"Communication skill coefficient: {w[2, 0]:.4f}"
    )

    print(f"\nMinimized Error: {error:.4f}")

    os.makedirs("reports/figures", exist_ok=True)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        X[:, 0],
        X[:, 1],
        y.ravel(),
        alpha=0.4
    )

    x1 = np.linspace(
        X[:, 0].min(),
        X[:, 0].max(),
        30
    )

    x2 = np.linspace(
        X[:, 1].min(),
        X[:, 1].max(),
        30
    )

    x1_grid, x2_grid = np.meshgrid(x1, x2)

    y_grid = (
        w[0, 0]
        + w[1, 0] * x1_grid
        + w[2, 0] * x2_grid
    )

    ax.plot_surface(
        x1_grid,
        x2_grid,
        y_grid,
        alpha=0.5
    )

    ax.set_xlabel("CGPA")
    ax.set_ylabel("Communication Skill Score")
    ax.set_zlabel("Salary Package (LPA)")
    ax.set_title("Linear Regression - Least Squares")

    plt.tight_layout()
    plt.savefig(
        "reports/figures/linear_regression_plane.png",
        dpi=150
    )
    plt.close()

    print(
        "\nSaved: reports/figures/linear_regression_plane.png"
    )


if __name__ == "__main__":
    train_linear_regression_ls()