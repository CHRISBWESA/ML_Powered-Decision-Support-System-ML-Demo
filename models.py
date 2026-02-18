"""
models.py — ML Logic Layer
══════════════════════════
Contains all machine learning functionality:
- Data preprocessing
- Model definitions with hyperparameters
- Training and evaluation
- Metric computation
- Visualization generation

Used by app.py (the Streamlit dashboard).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_curve, auc,
    mean_squared_error, r2_score, classification_report
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression


# ─────────────────────────────────────────────────────────────
# DARK PLOT STYLE
# ─────────────────────────────────────────────────────────────

def set_dark_fig_style():
    """Apply a consistent dark theme to all matplotlib figures."""
    plt.rcParams.update({
        "figure.facecolor": "#131720",
        "axes.facecolor":   "#0d0f14",
        "axes.edgecolor":   "#1e2533",
        "axes.labelcolor":  "#94a3b8",
        "xtick.color":      "#64748b",
        "ytick.color":      "#64748b",
        "text.color":       "#e2e8f0",
        "grid.color":       "#1e2533",
        "grid.linestyle":   "--",
        "grid.alpha":        0.6,
        "font.family":      "monospace",
        "axes.titlecolor":  "#e2e8f0",
    })


# ─────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────

def preprocess_data(df: pd.DataFrame, target_col: str, task_type: str):
    """
    Clean and prepare data for training.

    Steps:
      1. Drop rows with missing target values
      2. Label-encode all categorical feature columns
      3. Median-impute remaining numeric NaNs
      4. Train/test split (80/20)
      5. StandardScaler fit on train, applied to both splits
      6. Encode target for classification if it's a string column

    Returns:
        X_train, X_test, y_train, y_test,
        scaler, encoders (dict), label_encoder, feature_cols (list)
    """
    df = df.dropna(subset=[target_col]).copy()

    # Encode categorical feature columns
    encoders = {}
    for col in df.select_dtypes(include=["object", "category"]).columns:
        if col != target_col:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    # Fill remaining numeric NaNs
    df = df.fillna(df.median(numeric_only=True))

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode string target for classification
    label_encoder = None
    if task_type == "Classification" and y.dtype == object:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    return (
        X_train_scaled, X_test_scaled,
        y_train, y_test,
        scaler, encoders, label_encoder,
        list(X.columns)
    )


def preprocess_prediction_input(pred_df: pd.DataFrame, feature_cols: list,
                                 encoders: dict, scaler: StandardScaler) -> np.ndarray:
    """
    Prepare user-uploaded prediction data using the saved preprocessing artifacts.

    - Applies saved label encoders to categorical columns
    - Fills NaNs with column medians
    - Aligns columns to training feature set (fills missing cols with 0)
    - Applies saved scaler

    Returns:
        Scaled numpy array ready for model.predict()
    """
    df = pred_df.copy()

    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col].astype(str))

    df = df.fillna(df.median(numeric_only=True))

    # Fill any columns missing from prediction file
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    return scaler.transform(df[feature_cols])


# ─────────────────────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────

def get_classification_models(params: dict) -> dict:
    """
    Build classification models from hyperparameter dict.

    Expected keys:
        knn_k, svm_C, svm_kernel,
        rf_n, rf_depth, dt_depth
    """
    return {
        "KNN": KNeighborsClassifier(
            n_neighbors=params["knn_k"]
        ),
        "SVM": SVC(
            C=params["svm_C"],
            kernel=params["svm_kernel"],
            probability=True
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=params["rf_n"],
            max_depth=params["rf_depth"],
            random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=params["dt_depth"],
            random_state=42
        ),
    }


def get_regression_models(params: dict) -> dict:
    """
    Build regression models from hyperparameter dict.

    Expected keys:
        svr_C, svr_kernel, svr_eps,
        rf_n, rf_depth
    """
    return {
        "Linear Regression": LinearRegression(),
        "SVR": SVR(
            C=params["svr_C"],
            kernel=params["svr_kernel"],
            epsilon=params["svr_eps"]
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=params["rf_n"],
            max_depth=params["rf_depth"],
            random_state=42
        ),
    }


# ─────────────────────────────────────────────────────────────
# TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────────

def train_all_models(models: dict, X_train, y_train,
                     X_test, y_test, task_type: str):
    """
    Fit every model in the dict and compute evaluation metrics.

    Classification metrics: Accuracy
    Regression metrics:     MSE, RMSE, R²

    Returns:
        trained_models (dict), metrics (dict)
    """
    trained, metrics = {}, {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        trained[name] = model

        if task_type == "Classification":
            metrics[name] = {
                "Accuracy": round(accuracy_score(y_test, y_pred), 4)
            }
        else:
            mse = mean_squared_error(y_test, y_pred)
            metrics[name] = {
                "MSE":  round(mse, 4),
                "RMSE": round(np.sqrt(mse), 4),
                "R²":   round(r2_score(y_test, y_pred), 4),
            }

    return trained, metrics


def get_classification_report(model, X_test, y_test) -> pd.DataFrame:
    """Return a DataFrame of precision/recall/F1 per class."""
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return pd.DataFrame(report).T


# ─────────────────────────────────────────────────────────────
# VISUALIZATIONS — CLASSIFICATION
# ─────────────────────────────────────────────────────────────

def plot_accuracy_comparison(metrics: dict):
    """Bar chart comparing accuracy across all classification models."""
    set_dark_fig_style()
    names  = list(metrics.keys())
    accs   = [metrics[n]["Accuracy"] for n in names]
    colors = ["#38bdf8", "#818cf8", "#f472b6", "#34d399"]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(names, accs, color=colors[:len(names)],
                  edgecolor="#0d0f14", linewidth=1.5)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_title("Model Accuracy Comparison", fontsize=12, pad=12)
    ax.axhline(y=max(accs), color="#fbbf24", linestyle="--", alpha=0.5, linewidth=1)

    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{acc:.3f}", ha="center", va="bottom",
            fontsize=9, color="#e2e8f0"
        )
    ax.grid(axis="y")
    plt.tight_layout()
    return fig


def plot_confusion_matrices(trained_models: dict, X_test, y_test):
    """Side-by-side confusion matrix heatmaps for every trained model."""
    set_dark_fig_style()
    n    = len(trained_models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, trained_models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    cbar=False, linewidths=0.5, linecolor="#1e2533")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    return fig


def plot_roc_curves(trained_models: dict, X_test, y_test):
    """
    ROC curves for binary classification only.
    Returns None if target has more than 2 classes.
    """
    if len(np.unique(y_test)) != 2:
        return None

    set_dark_fig_style()
    colors = ["#38bdf8", "#818cf8", "#f472b6", "#34d399"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], "w--", alpha=0.3, linewidth=1)

    for (name, model), color in zip(trained_models.items(), colors):
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
        else:
            continue
        fpr, tpr, _ = roc_curve(y_test, scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{name} (AUC={roc_auc:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves", fontsize=12)
    legend = ax.legend(fontsize=9, framealpha=0.1)
    for text in legend.get_texts():
        text.set_color("#e2e8f0")
    ax.grid(True)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# VISUALIZATIONS — REGRESSION
# ─────────────────────────────────────────────────────────────

def plot_regression_scatter(trained_models: dict, X_test, y_test):
    """Actual vs. predicted scatter plots for every regression model."""
    set_dark_fig_style()
    n      = len(trained_models)
    colors = ["#38bdf8", "#818cf8", "#34d399"]
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, model), color in zip(axes, trained_models.items(), colors):
        y_pred = model.predict(X_test)
        ax.scatter(y_test, y_pred, alpha=0.6, color=color, s=30, edgecolors="none")
        mn = min(float(np.array(y_test).min()), float(y_pred.min()))
        mx = max(float(np.array(y_test).max()), float(y_pred.max()))
        ax.plot([mn, mx], [mn, mx], "w--", alpha=0.4, linewidth=1)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")

    plt.tight_layout()
    return fig


def plot_regression_metrics(metrics: dict):
    """Side-by-side MSE and R² bar charts for all regression models."""
    set_dark_fig_style()
    names     = list(metrics.keys())
    mse_vals  = [metrics[n]["MSE"] for n in names]
    r2_vals   = [metrics[n]["R²"]  for n in names]
    colors    = ["#38bdf8", "#818cf8", "#34d399"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.bar(names, mse_vals, color=colors[:len(names)], edgecolor="#0d0f14")
    ax1.set_title("Mean Squared Error", fontsize=11)
    ax1.set_ylabel("MSE")
    ax1.grid(axis="y")

    ax2.bar(names, r2_vals, color=colors[:len(names)], edgecolor="#0d0f14")
    ax2.set_title("R² Score", fontsize=11)
    ax2.set_ylabel("R²")
    ax2.set_ylim(min(0, min(r2_vals)) - 0.05, 1.05)
    ax2.grid(axis="y")

    plt.tight_layout()
    return fig
