from __future__ import annotations

## imports##

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt


# =========================
# Helpers
# =========================

def add_bias_column(X: np.ndarray) -> np.ndarray:
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    return np.hstack([np.ones((X.shape[0], 1)), X])


def standardize_train_test(
    X_train: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0, ddof=0)
    sigma = np.where(sigma == 0, 1.0, sigma)

    return (X_train - mu) / sigma, (X_test - mu) / sigma, mu, sigma


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    return float(np.mean((y_true - y_pred) ** 2))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0

    return float(1.0 - ss_res / ss_tot)


@dataclass
class GDResult:
    theta: np.ndarray
    losses: np.ndarray
    thetas: np.ndarray
    
# =========================
# Q1 Gradient Descent & Visualization
# =========================

def gradient_descent_linreg(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.05,
    epochs: int = 200,
    theta0: Optional[np.ndarray] = None,
) -> GDResult:

    n, d = X.shape

    if theta0 is None:
        theta = np.zeros(d)
    else:
        theta = theta0.copy()

    losses = np.zeros(epochs)
    theta_path = np.zeros((epochs, d))

    for i in range(epochs):

        y_pred = X @ theta
        error = y_pred - y

        losses[i] = np.mean(error ** 2)

        gradient = (2 / n) * (X.T @ error)

        theta = theta - lr * gradient

        theta_path[i] = theta

    return GDResult(theta=theta, losses=losses, thetas=theta_path)

def visualize_gradient_descent(
    lr: float = 0.1,
    epochs: int = 60,
    seed: int = 0,
) -> Dict[str, np.ndarray]:

    np.random.seed(seed)

    n = 60
    X_feature = np.random.randn(n, 1)

    true_theta0 = 1.5
    true_theta1 = 2.5

    noise = np.random.randn(n) * 0.3
    y = true_theta0 + true_theta1 * X_feature[:, 0] + noise

    X = add_bias_column(X_feature)

    result = gradient_descent_linreg(X, y, lr=lr, epochs=epochs)

    return {
        "theta_path": result.thetas,
        "losses": result.losses,
        "X": X,
        "y": y,
    }

# =========================
# Train/Test split helper
# =========================

def train_test_split_np(X, y, test_size=0.2, seed=0):

    np.random.seed(seed)

    n = X.shape[0]
    indices = np.random.permutation(n)

    test_n = int(n * test_size)

    test_idx = indices[:test_n]
    train_idx = indices[test_n:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# =========================
# Q2 Diabetes using GD
# =========================

def diabetes_linear_gd(
    lr: float = 0.05,
    epochs: int = 2000,
    test_size: float = 0.2,
    seed: int = 0,
) -> Tuple[float, float, float, float, np.ndarray]:

    data = datasets.load_diabetes()

    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split_np(
        X, y, test_size=test_size, seed=seed
    )

    X_train, X_test, _, _ = standardize_train_test(X_train, X_test)

    X_train = add_bias_column(X_train)
    X_test = add_bias_column(X_test)

    res = gradient_descent_linreg(X_train, y_train, lr=lr, epochs=epochs)

    theta = res.theta

    y_train_pred = X_train @ theta
    y_test_pred = X_test @ theta

    train_mse = mse(y_train, y_train_pred)
    test_mse = mse(y_test, y_test_pred)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    return train_mse, test_mse, train_r2, test_r2, theta


# =========================
# Q3 Analytical solution
# =========================

def diabetes_linear_analytical(
    ridge_lambda: float = 1e-8,
    test_size: float = 0.2,
    seed: int = 0,
) -> Tuple[float, float, float, float, np.ndarray]:

    data = datasets.load_diabetes()

    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split_np(
        X, y, test_size=test_size, seed=seed
    )

    X_train, X_test, _, _ = standardize_train_test(X_train, X_test)

    X_train = add_bias_column(X_train)
    X_test = add_bias_column(X_test)

    d = X_train.shape[1]

    I = np.eye(d)

    theta = np.linalg.solve(
        X_train.T @ X_train + ridge_lambda * I,
        X_train.T @ y_train,
    )

    y_train_pred = X_train @ theta
    y_test_pred = X_test @ theta

    train_mse = mse(y_train, y_train_pred)
    test_mse = mse(y_test, y_test_pred)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    return train_mse, test_mse, train_r2, test_r2, theta


# =========================
# Q4 Compare GD vs Analytical
# =========================

def diabetes_compare_gd_vs_analytical(
    lr: float = 0.05,
    epochs: int = 4000,
    test_size: float = 0.2,
    seed: int = 0,
) -> Dict[str, float]:

    gd = diabetes_linear_gd(lr, epochs, test_size, seed)
    an = diabetes_linear_analytical(1e-8, test_size, seed)

    train_mse_gd, test_mse_gd, train_r2_gd, test_r2_gd, theta_gd = gd
    train_mse_an, test_mse_an, train_r2_an, test_r2_an, theta_an = an

    theta_l2_diff = np.linalg.norm(theta_gd - theta_an)

    theta_cosine_sim = np.dot(theta_gd, theta_an) / (
        np.linalg.norm(theta_gd) * np.linalg.norm(theta_an)
    )

    return {
        "theta_l2_diff": float(theta_l2_diff),
        "train_mse_diff": float(abs(train_mse_gd - train_mse_an)),
        "test_mse_diff": float(abs(test_mse_gd - test_mse_an)),
        "train_r2_diff": float(abs(train_r2_gd - train_r2_an)),
        "test_r2_diff": float(abs(test_r2_gd - test_r2_an)),
        "theta_cosine_sim": float(theta_cosine_sim),
    }

#================
# Visualization
#================
result = visualize_gradient_descent(lr=0.1, epochs=60, seed=0)

theta_path = result["theta_path"]
losses = result["losses"]
X = result["X"]
y = result["y"]

# =========================
# Plot 1: Loss vs Epoch
# =========================

plt.figure()
plt.plot(losses)
plt.title("Gradient Descent Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.show()


# =========================
# Plot 2: Data + Learned Line
# =========================

plt.figure()

# original data
plt.scatter(X[:,1], y)

# learned line
theta_final = theta_path[-1]
x_line = X[:,1]
y_line = theta_final[0] + theta_final[1] * x_line

plt.plot(x_line, y_line)

plt.title("Linear Regression Fit")
plt.xlabel("X")
plt.ylabel("y")
plt.show()


# =========================
# Plot 3: Parameter Path
# =========================

plt.figure()

theta0 = theta_path[:,0]
theta1 = theta_path[:,1]

plt.plot(theta0, theta1, marker='o')
plt.title("Gradient Descent Parameter Path")
plt.xlabel("Theta 0")
plt.ylabel("Theta 1")

plt.show()
