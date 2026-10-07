import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


np.random.seed(42)

X = np.sort(6 * np.random.rand(100, 1) + 4)
y = np.sin(X).ravel() + np.random.normal(
    0, 0.2, X.shape[0]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

poly_degree = 15

poly = PolynomialFeatures(
    degree=poly_degree
)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)

lambdas = np.logspace(-4, 4, 200)

train_errors = []
test_errors = []

for lam in lambdas:

    ridge = Ridge(alpha=lam)

    ridge.fit(
        X_train_scaled,
        y_train
    )

    y_train_pred = ridge.predict(
        X_train_scaled
    )

    y_test_pred = ridge.predict(
        X_test_scaled
    )

    train_errors.append(
        mean_squared_error(
            y_train,
            y_train_pred
        )
    )

    test_errors.append(
        mean_squared_error(
            y_test,
            y_test_pred
        )
    )


best_index = np.argmin(test_errors)
best_lambda = lambdas[best_index]

print("\n" + "=" * 50)
print("--- RIDGE / L2 POLYNOMIAL REGRESSION ---")
print(f"Polynomial Degree: {poly_degree}")
print(f"Best Lambda: {best_lambda:.6f}")
print(
    f"Best Test MSE: "
    f"{test_errors[best_index]:.6f}"
)

plt.figure(figsize=(10, 6))

plt.plot(
    lambdas,
    train_errors,
    label="Training Error"
)

plt.plot(
    lambdas,
    test_errors,
    label="Testing Error"
)

plt.xscale("log")

plt.xlabel("Regularization Parameter (Lambda)")
plt.ylabel("Mean Squared Error")
plt.title(
    "Ridge Regression Regularization Path"
)

plt.legend()
plt.grid(True, which="both", linestyle="--")
plt.tight_layout()

plt.savefig(
    "reports/figures/ridge_regularization_path.png",
    dpi=150
)

plt.close()

print(
    "Saved: reports/figures/ridge_regularization_path.png"
)