"""
models.py — ML Logic Layer
══════════════════════════
All machine learning functionality used by app.py:
  - Robust data preprocessing & encoding
  - Model definitions with hyperparameters
  - Training and evaluation
  - Single-row prediction helper
  - Metric explanations
  - Visualization generation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_curve, auc,
    mean_squared_error, r2_score, classification_report,
    precision_score, recall_score, f1_score, mean_absolute_error
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression


# ─────────────────────────────────────────────────────────────
# PLOT STYLE
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
# ENCODING HELPERS
# ─────────────────────────────────────────────────────────────

def _is_categorical_col(series: pd.Series) -> bool:
    """
    Returns True if a column should be label-encoded.
    Handles object, string, category, and bool dtypes.
    Does NOT treat numeric columns as categorical.
    """
    if pd.api.types.is_bool_dtype(series):
        return True
    if pd.api.types.is_categorical_dtype(series):
        return True
    if pd.api.types.is_object_dtype(series):
        return True
    # pandas 2.x StringDtype — is_string_dtype returns True for object too,
    # so we guard with is_numeric_dtype to avoid double-encoding numerics.
    if pd.api.types.is_string_dtype(series) and not pd.api.types.is_numeric_dtype(series):
        return True
    return False


def _safe_label_encode(series: pd.Series, le: LabelEncoder) -> pd.Series:
    """
    Apply a fitted LabelEncoder to a series.
    Unseen labels (not in le.classes_) are mapped to the first known class
    instead of raising ValueError — critical for robust prediction on new data.
    """
    known    = set(le.classes_)
    fallback = le.classes_[0]
    cleaned  = series.astype(str).map(lambda v: v if v in known else fallback)
    return pd.Series(le.transform(cleaned), index=series.index)


def build_encoders(df: pd.DataFrame, target_col: str) -> dict:
    """
    Fit one LabelEncoder per categorical feature column (excluding target).
    Stores the class list alongside each encoder so the UI can render
    proper dropdown widgets.

    Returns:
        { col_name: {"le": LabelEncoder, "classes": [str, ...]} }
    """
    encoders = {}
    for col in df.columns:
        if col == target_col:
            continue
        if _is_categorical_col(df[col]):
            le = LabelEncoder()
            le.fit(df[col].fillna("__missing__").astype(str))
            encoders[col] = {"le": le, "classes": list(le.classes_)}
    return encoders


def encode_features(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """
    Apply fitted encoders to a copy of the dataframe.
    NaN values are treated as the string '__missing__' before encoding.
    Unseen labels are gracefully handled (mapped to most-frequent class).
    """
    df = df.copy()
    for col, enc in encoders.items():
        if col in df.columns:
            df[col] = df[col].fillna("__missing__").astype(str)
            df[col] = _safe_label_encode(df[col], enc["le"])
    return df


def encode_target(series: pd.Series, task_type: str):
    """
    Encode the target column for training.

      Classification + non-numeric  → LabelEncoder → (encoded_series, le)
      Classification + numeric      → cast to int   → (series, None)
      Regression                    → cast to float → (series, None)
    """
    if task_type == "Classification":
        if not pd.api.types.is_numeric_dtype(series):
            le = LabelEncoder()
            encoded = pd.Series(
                le.fit_transform(series.fillna("__missing__").astype(str)),
                index=series.index
            )
            return encoded, le
        else:
            return series.astype(int), None
    else:
        return pd.to_numeric(series, errors="coerce"), None


# ─────────────────────────────────────────────────────────────
# PREPROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────

def preprocess_data(df: pd.DataFrame, target_col: str, task_type: str):
    """
    Full preprocessing pipeline:

      1. Drop rows where target is missing
      2. Build & apply feature encoders  (categorical → integer)
      3. Encode target column
      4. Coerce features to numeric, median-impute remaining NaNs
      5. Stratified train/test split  (80/20)
      6. Fit StandardScaler on train, apply to both splits

    Returns:
        X_train_scaled, X_test_scaled,
        y_train, y_test,
        scaler, encoders, label_encoder,
        feature_cols  (list),
        col_meta      (dict: col → metadata for building prediction UI widgets)
    """
    df = df.dropna(subset=[target_col]).copy()

    # ── Feature encoding ─────────────────────────────────────
    encoders = build_encoders(df, target_col)
    df       = encode_features(df, encoders)

    # ── Target encoding ──────────────────────────────────────
    y, label_encoder = encode_target(df[target_col], task_type)

    # ── Feature matrix ───────────────────────────────────────
    X = df.drop(columns=[target_col])
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    feature_cols = list(X.columns)

    # ── Column metadata for prediction UI ────────────────────
    # Computed BEFORE scaling so ranges are in human-readable units.
    orig_X   = df.drop(columns=[target_col])
    col_meta = {}
    for col in feature_cols:
        if col in encoders:
            # Categorical: expose original class labels for a selectbox
            classes_no_missing = [c for c in encoders[col]["classes"] if c != "__missing__"]
            col_meta[col] = {
                "type":    "categorical",
                "values":  classes_no_missing or encoders[col]["classes"],
            }
        else:
            col_data = pd.to_numeric(orig_X[col], errors="coerce").dropna()
            col_meta[col] = {
                "type":   "numeric",
                "min":    float(col_data.min()),
                "max":    float(col_data.max()),
                "median": float(col_data.median()),
            }

    # ── Train/test split ─────────────────────────────────────
    try:
        stratify = y if task_type == "Classification" else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )
    except ValueError:
        # stratify fails when some class has only 1 sample; fall back
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    # ── Scaling ──────────────────────────────────────────────
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    return (
        X_train_scaled, X_test_scaled,
        y_train, y_test,
        scaler, encoders, label_encoder,
        feature_cols, col_meta
    )


# ─────────────────────────────────────────────────────────────
# SINGLE-ROW PREDICTION
# ─────────────────────────────────────────────────────────────

def predict_single_row(input_dict: dict, feature_cols: list,
                        encoders: dict, scaler: StandardScaler,
                        model, label_encoder):
    """
    Predict for one sample entered manually in the UI.

    input_dict: { feature_name: raw_value_from_widget }
    Returns the prediction as a human-readable string.
    """
    # Build one-row DataFrame in the correct column order
    row = pd.DataFrame([{col: input_dict.get(col, np.nan) for col in feature_cols}])

    # Encode categorical columns using saved encoders
    for col, enc in encoders.items():
        if col in row.columns:
            row[col] = row[col].fillna("__missing__").astype(str)
            row[col] = _safe_label_encode(row[col], enc["le"])

    # Ensure all values are numeric; fill any remaining NaN with 0
    row = row.apply(pd.to_numeric, errors="coerce").fillna(0)

    row_scaled = scaler.transform(row[feature_cols])
    pred       = model.predict(row_scaled)

    # Decode label back to original string if applicable
    if label_encoder is not None:
        pred = label_encoder.inverse_transform(pred)

    return str(pred[0])


# ─────────────────────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────

def get_classification_models(params: dict) -> dict:
    """
    Build classification model instances from a hyperparameter dict.
    Expected keys: knn_k, svm_C, svm_kernel, rf_n, rf_depth, dt_depth
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
    Build regression model instances from a hyperparameter dict.
    Expected keys: svr_C, svr_kernel, svr_eps, rf_n, rf_depth
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

def train_all_models(model_dict: dict, X_train, y_train,
                     X_test, y_test, task_type: str):
    """
    Fit every model in model_dict and compute evaluation metrics.

    Classification → Accuracy, Precision, Recall, F1  (macro avg)
    Regression     → R², MSE, RMSE, MAE
    """
    trained, metrics = {}, {}

    for name, model in model_dict.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        trained[name] = model

        if task_type == "Classification":
            avg = "binary" if len(np.unique(y_test)) == 2 else "macro"
            metrics[name] = {
                "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
                "Precision": round(precision_score(y_test, y_pred, average=avg, zero_division=0), 4),
                "Recall":    round(recall_score(y_test, y_pred, average=avg, zero_division=0), 4),
                "F1":        round(f1_score(y_test, y_pred, average=avg, zero_division=0), 4),
            }
        else:
            mse = mean_squared_error(y_test, y_pred)
            metrics[name] = {
                "R²":   round(r2_score(y_test, y_pred), 4),
                "MSE":  round(mse, 4),
                "RMSE": round(np.sqrt(mse), 4),
                "MAE":  round(mean_absolute_error(y_test, y_pred), 4),
            }

    return trained, metrics


def get_classification_report(model, X_test, y_test) -> pd.DataFrame:
    """Return precision / recall / F1 per class as a DataFrame."""
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return pd.DataFrame(report).T


# ─────────────────────────────────────────────────────────────
# METRIC EXPLANATIONS  (used by the Evaluate tab in app.py)
# ─────────────────────────────────────────────────────────────

CLASSIFICATION_METRIC_INFO = {
    "Accuracy": {
        "icon":    "🎯",
        "formula": "Correct predictions ÷ Total predictions",
        "range":   "0 → 1  (higher is better)",
        "plain": (
            "The fraction of samples the model labelled correctly. "
            "Simple and intuitive, but misleading on imbalanced datasets — "
            "a model that always predicts the majority class can still look good."
        ),
        "when": "Quick gut-check. Supplement with F1 when classes are unbalanced.",
    },
    "Precision": {
        "icon":    "🔬",
        "formula": "True Positives ÷ (True Positives + False Positives)",
        "range":   "0 → 1  (higher is better)",
        "plain": (
            "Of everything the model flagged as positive, how many actually were? "
            "High precision = few false alarms. "
            "Critical when acting on a false positive is costly (e.g. spam filters, fraud alerts)."
        ),
        "when": "Prioritise when false positives are expensive.",
    },
    "Recall": {
        "icon":    "🕵️",
        "formula": "True Positives ÷ (True Positives + False Negatives)",
        "range":   "0 → 1  (higher is better)",
        "plain": (
            "Of all actual positives in the data, how many did the model catch? "
            "High recall = few missed cases. "
            "Critical when missing a real positive is dangerous (e.g. cancer screening, fraud detection)."
        ),
        "when": "Prioritise when false negatives are expensive.",
    },
    "F1": {
        "icon":    "⚖️",
        "formula": "2 × (Precision × Recall) ÷ (Precision + Recall)",
        "range":   "0 → 1  (higher is better)",
        "plain": (
            "The harmonic mean of Precision and Recall — a single balanced score. "
            "It punishes models that sacrifice one metric to inflate the other. "
            "The best all-round metric for imbalanced datasets."
        ),
        "when": "Default choice when class sizes differ significantly.",
    },
}

REGRESSION_METRIC_INFO = {
    "R²": {
        "icon":    "📐",
        "formula": "1 − (Σ(actual−predicted)²) ÷ (Σ(actual−mean)²)",
        "range":   "−∞ → 1  (closer to 1 is better)",
        "plain": (
            "Proportion of the target's variance explained by the model. "
            "R²=1 → perfect; R²=0 → no better than predicting the mean; "
            "negative → actively worse than the mean."
        ),
        "when": "Start here — gives the clearest picture of overall fit.",
    },
    "MSE": {
        "icon":    "📏",
        "formula": "mean( (actual − predicted)² )",
        "range":   "0 → ∞  (lower is better)",
        "plain": (
            "Average squared error. Squaring amplifies large mistakes, "
            "so MSE is very sensitive to outliers. "
            "The units are squared, making direct interpretation harder."
        ),
        "when": "Use when big errors must be heavily penalised.",
    },
    "RMSE": {
        "icon":    "📉",
        "formula": "√MSE",
        "range":   "0 → ∞  (lower is better)",
        "plain": (
            "Square root of MSE — brings the error back to the same units as the target. "
            "An RMSE of 5 means predictions are roughly ±5 off on average "
            "(with outliers weighted more than in MAE)."
        ),
        "when": "Most interpretable error — use alongside R².",
    },
    "MAE": {
        "icon":    "🧮",
        "formula": "mean( |actual − predicted| )",
        "range":   "0 → ∞  (lower is better)",
        "plain": (
            "Average absolute error — treats all mistakes equally, "
            "making it more robust to outliers than RMSE. "
            "Directly interpretable in the target's units."
        ),
        "when": "Use when your data has outliers and you want a robust error estimate.",
    },
}


# ─────────────────────────────────────────────────────────────
# VISUALIZATIONS — CLASSIFICATION
# ─────────────────────────────────────────────────────────────

def plot_accuracy_comparison(metrics: dict):
    """Grouped bar chart showing Accuracy, Precision, Recall, F1 per model."""
    set_dark_fig_style()
    names        = list(metrics.keys())
    metric_keys  = ["Accuracy", "Precision", "Recall", "F1"]
    colors       = ["#38bdf8", "#818cf8", "#f472b6", "#34d399"]

    x     = np.arange(len(names))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (mk, color) in enumerate(zip(metric_keys, colors)):
        vals = [metrics[n].get(mk, 0) for n in names]
        bars = ax.bar(x + i * width, vals, width, label=mk,
                      color=color, edgecolor="#0d0f14", linewidth=1)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7, color="#e2e8f0")

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontsize=12, pad=12)
    legend = ax.legend(fontsize=9, framealpha=0.15)
    for t in legend.get_texts():
        t.set_color("#e2e8f0")
    ax.grid(axis="y")
    plt.tight_layout()
    return fig


def plot_confusion_matrices(trained_models: dict, X_test, y_test):
    """Side-by-side confusion matrix heatmaps for every trained model."""
    set_dark_fig_style()
    n = len(trained_models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, trained_models.items()):
        y_pred = model.predict(X_test)
        cm     = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    cbar=False, linewidths=0.5, linecolor="#1e2533")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    return fig


def plot_roc_curves(trained_models: dict, X_test, y_test):
    """ROC curves for binary classification only. Returns None for multiclass."""
    if len(np.unique(y_test)) != 2:
        return None

    set_dark_fig_style()
    colors = ["#38bdf8", "#818cf8", "#f472b6", "#34d399"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], "w--", alpha=0.3, linewidth=1, label="Random (AUC=0.500)")

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
    for t in legend.get_texts():
        t.set_color("#e2e8f0")
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
    """Grouped bar chart for R², RMSE, MAE across all regression models."""
    set_dark_fig_style()
    names        = list(metrics.keys())
    metric_keys  = ["R²", "RMSE", "MAE"]
    colors       = ["#34d399", "#f472b6", "#818cf8"]

    x     = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 4))

    for i, (mk, color) in enumerate(zip(metric_keys, colors)):
        vals = [metrics[n].get(mk, 0) for n in names]
        bars = ax.bar(x + i * width, vals, width, label=mk,
                      color=color, edgecolor="#0d0f14")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=8, color="#e2e8f0")

    ax.set_xticks(x + width)
    ax.set_xticklabels(names)
    ax.set_title("Regression Metrics Comparison", fontsize=11)
    ax.grid(axis="y")
    legend = ax.legend(fontsize=9, framealpha=0.15)
    for t in legend.get_texts():
        t.set_color("#e2e8f0")
    plt.tight_layout()
    return fig