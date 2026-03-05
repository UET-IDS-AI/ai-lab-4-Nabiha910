"""
AI_stats_lab.py

You must implement the TODO functions below.
Do not change function names or return signatures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn import datasets


# =========================
# Helpers (you may use these)
# =========================

def add_bias_column(X: np.ndarray) -> np.ndarray:
    """Add a bias (intercept) column of ones to X."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    return np.hstack([np.ones((X.shape[0], 1)), X])


def standardize_train_test(
    X_train: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize using train statistics only.
    Returns: X_train_std, X_test_std, mean, std
    """

    # compute mean of training data
    mu = X_train.mean(axis=0)

    # compute standard deviation
    sigma = X_train.std(axis=0, ddof=0)

    # avoid divide by zero
    sigma = np.where(sigma == 0, 1.0, sigma)

    # standardize both datasets using train stats
    return (X_train - mu) / sigma, (X_test - mu) / sigma, mu, sigma


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error"""
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    return float(np.mean((y_true - y_pred) ** 2))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² Score"""

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
# Q1: Gradient descent + visualization data
# =========================

def gradient_descent_linreg(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.05,
    epochs: int = 200,
    theta0: Optional[np.ndarray] = None,
) -> GDResult:
    """
    Linear regression with batch gradient descent using MSE loss
    """

    # number of training samples and parameters
    n, d = X.shape

    # initialize parameters
    if theta0 is None:
        theta = np.zeros(d)
    else:
        theta = theta0.copy()

    losses = []
    theta_history = []

    for _ in range(epochs):

        # prediction
        y_pred = X @ theta

        # error
        error = y_pred - y

        # gradient of MSE loss
        gradient = (2 / n) * (X.T @ error)

        # parameter update
        theta = theta - lr * gradient

        # compute loss
        loss = np.mean(error ** 2)

        losses.append(loss)

        # save theta trajectory
        theta_history.append(theta.copy())

    return GDResult(
        theta=np.array(theta),
        losses=np.array(losses),
        thetas=np.array(theta_history),
    )


def visualize_gradient_descent(
    lr: float = 0.1,
    epochs: int = 60,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Create synthetic dataset for visualization
    """

    np.random.seed(seed)

    n = 100

    # generate random feature
    X_feature = np.random.randn(n, 1)

    # true model parameters
    theta0_true = 2
    theta1_true = 3

    # noise
    noise = np.random.randn(n) * 0.5

    # generate target
    y = theta0_true + theta1_true * X_feature[:, 0] + noise

    # add bias column
    X = add_bias_column(X_feature)

    # run gradient descent
    result = gradient_descent_linreg(X, y, lr=lr, epochs=epochs)

    return {
        "theta_path": result.thetas,
        "losses": result.losses,
        "X": X,
        "y": y,
    }


# =========================
# Q2: Diabetes regression using gradient descent
# =========================

def diabetes_linear_gd(
    lr: float = 0.05,
    epochs: int = 2000,
    test_size: float = 0.2,
    seed: int = 0,
) -> Tuple[float, float, float, float, np.ndarray]:

    # load dataset
    data = datasets.load_diabetes()

    X = data.data
    y = data.target

    # random split
    np.random.seed(seed)

    n = len(X)

    indices = np.random.permutation(n)

    test_n = int(n * test_size)

    test_idx = indices[:test_n]
    train_idx = indices[test_n:]

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    # standardize
    X_train, X_test, _, _ = standardize_train_test(X_train, X_test)

    # add bias
    X_train = add_bias_column(X_train)
    X_test = add_bias_column(X_test)

    # train model
    result = gradient_descent_linreg(X_train, y_train, lr=lr, epochs=epochs)

    theta = result.theta

    # predictions
    train_pred = X_train @ theta
    test_pred = X_test @ theta

    # evaluation
    train_mse = mse(y_train, train_pred)
    test_mse = mse(y_test, test_pred)

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    return train_mse, test_mse, train_r2, test_r2, theta


# =========================
# Q3: Analytical solution
# =========================

def diabetes_linear_analytical(
    ridge_lambda: float = 1e-8,
    test_size: float = 0.2,
    seed: int = 0,
) -> Tuple[float, float, float, float, np.ndarray]:

    # load dataset
    data = datasets.load_diabetes()

    X = data.data
    y = data.target

    # random split
    np.random.seed(seed)

    n = len(X)

    indices = np.random.permutation(n)

    test_n = int(n * test_size)

    test_idx = indices[:test_n]
    train_idx = indices[test_n:]

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    # standardization
    X_train, X_test, _, _ = standardize_train_test(X_train, X_test)

    # add bias
    X_train = add_bias_column(X_train)
    X_test = add_bias_column(X_test)

    # normal equation
    d = X_train.shape[1]

    I = np.eye(d)

    theta = np.linalg.inv(X_train.T @ X_train + ridge_lambda * I) @ X_train.T @ y_train

    # predictions
    train_pred = X_train @ theta
    test_pred = X_test @ theta

    # metrics
    train_mse = mse(y_train, train_pred)
    test_mse = mse(y_test, test_pred)

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    return train_mse, test_mse, train_r2, test_r2, theta


# =========================
# Q4: Compare GD vs analytical
# =========================

def diabetes_compare_gd_vs_analytical(
    lr: float = 0.05,
    epochs: int = 4000,
    test_size: float = 0.2,
    seed: int = 0,
) -> Dict[str, float]:

    gd_results = diabetes_linear_gd(lr, epochs, test_size, seed)

    an_results = diabetes_linear_analytical(1e-8, test_size, seed)

    train_mse_gd, test_mse_gd, train_r2_gd, test_r2_gd, theta_gd = gd_results

    train_mse_an, test_mse_an, train_r2_an, test_r2_an, theta_an = an_results

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
