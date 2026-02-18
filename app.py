import warnings
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st  # pyright: ignore[reportMissingImports]

import models  # ← all ML logic lives here

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG & CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Studio",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0f14;
    color: #e2e8f0;
}
.stApp { background-color: #0d0f14; }
.block-container { padding: 2rem 3rem; max-width: 1400px; }

.hero-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    border-bottom: 1px solid #1e2533;
    margin-bottom: 2rem;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.8rem;
    font-weight: 700;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.hero-subtitle {
    font-size: 1rem;
    color: #64748b;
    margin-top: 0.4rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-weight: 300;
}
.metric-card {
    background: #131720;
    border: 1px solid #1e2d3d;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin: 0.4rem 0;
}
.model-badge {
    display: inline-block;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    color: #94a3b8;
    margin: 2px;
}
.stTabs [data-baseweb="tab-list"] {
    background: #131720;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #1e2533;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #64748b;
    background: transparent;
    border-radius: 7px;
    padding: 8px 20px;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: #1e293b !important;
    color: #38bdf8 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #6366f1);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 1px;
    padding: 0.6rem 2rem;
    font-weight: 700;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }
.stFileUploader > div {
    background: #131720;
    border: 2px dashed #1e2d3d;
    border-radius: 10px;
}
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 0.8rem;
    border-left: 3px solid #38bdf8;
    padding-left: 10px;
}
.status-success {
    background: #052e16;
    border: 1px solid #166534;
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    color: #4ade80;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
}
.status-info {
    background: #0c1a2e;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    color: #60a5fa;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "df":             None,
        "target":         None,
        "task_type":      None,
        "X_train":        None,
        "X_test":         None,
        "y_train":        None,
        "y_test":         None,
        "scaler":         None,
        "encoders":       {},
        "label_encoder":  None,
        "trained_models": {},
        "metrics":        {},
        "feature_cols":   [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def no_data_warning():
    st.markdown(
        '<div class="status-info">ℹ️  Please upload a dataset in the '
        '<b>Upload & Configure</b> tab first.</div>',
        unsafe_allow_html=True
    )

def no_models_warning():
    st.markdown(
        '<div class="status-info">ℹ️  Train models first in the '
        '<b>Train & Compare</b> tab.</div>',
        unsafe_allow_html=True
    )

def close_fig(fig):
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <p class="hero-title">⚗️ ML STUDIO</p>
    <p class="hero-subtitle">Train · Compare · Evaluate · Predict</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "  📁  UPLOAD & CONFIGURE  ",
    "  🧠  TRAIN & COMPARE  ",
    "  📊  EVALUATE  ",
    "  🔮  PREDICT  ",
])


# ═══════════════════════════════════════════════════════════════
# TAB 1 — UPLOAD & CONFIGURE
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-label">Dataset Upload</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop your CSV file here",
        type=["csv"],
        help="Upload a CSV. Categorical features are auto-encoded; NaNs are median-imputed."
    )

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.session_state["df"] = df

            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                st.markdown('<p class="section-label">Target Column</p>', unsafe_allow_html=True)
                target = st.selectbox("Target column", df.columns.tolist(), key="target_select")
                st.session_state["target"] = target

            with col2:
                st.markdown('<p class="section-label">Task Type</p>', unsafe_allow_html=True)
                # Simple auto-detect heuristic
                n_unique   = df[target].nunique()
                auto_task  = "Classification" if (df[target].dtype == object or n_unique <= 20) else "Regression"
                task = st.radio(
                    "Task type",
                    ["Classification", "Regression"],
                    index=0 if auto_task == "Classification" else 1,
                    horizontal=True
                )
                st.session_state["task_type"] = task

            with col3:
                st.markdown('<p class="section-label">Dataset Info</p>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="metric-card">
                    <b style="color:#38bdf8">{df.shape[0]}</b> rows<br>
                    <b style="color:#818cf8">{df.shape[1]}</b> columns<br>
                    <b style="color:#f472b6">{df.isnull().sum().sum()}</b> nulls
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown('<p class="section-label">Data Preview</p>', unsafe_allow_html=True)
            st.dataframe(df.head(10), use_container_width=True)

            # Target distribution chart
            st.markdown('<p class="section-label">Target Distribution</p>', unsafe_allow_html=True)
            models.set_dark_fig_style()
            fig_dist, ax_dist = plt.subplots(figsize=(8, 3))
            if task == "Classification":
                vc = df[target].value_counts()
                ax_dist.bar(vc.index.astype(str), vc.values,
                            color="#38bdf8", edgecolor="#0d0f14")
                ax_dist.set_title(f"Class Distribution — {target}", fontsize=11)
            else:
                ax_dist.hist(df[target].dropna(), bins=30,
                             color="#818cf8", edgecolor="#0d0f14")
                ax_dist.set_title(f"Value Distribution — {target}", fontsize=11)
            ax_dist.grid(axis="y")
            plt.tight_layout()
            st.pyplot(fig_dist, use_container_width=False)
            close_fig(fig_dist)

            st.markdown(
                '<div class="status-success">✓ Dataset loaded. '
                'Head to <b>Train & Compare</b> to build models.</div>',
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"Error reading file: {e}")

    else:
        st.markdown(
            '<div class="status-info">ℹ️  Upload a CSV to get started.<br>'
            'Missing values and categorical features are handled automatically.</div>',
            unsafe_allow_html=True
        )


# ═══════════════════════════════════════════════════════════════
# TAB 2 — TRAIN & COMPARE
# ═══════════════════════════════════════════════════════════════
with tab2:
    if st.session_state["df"] is None:
        no_data_warning()
    else:
        task   = st.session_state["task_type"]
        df     = st.session_state["df"]
        target = st.session_state["target"]

        st.markdown(
            f'<p class="section-label">Hyperparameters — {task}</p>',
            unsafe_allow_html=True
        )

        # ── Hyperparameter widgets ──────────────────────────────
        params = {}

        if task == "Classification":
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown("**KNN**")
                params["knn_k"] = st.slider("K (neighbors)", 1, 25, 5, key="knn_k")
            with c2:
                st.markdown("**SVM**")
                params["svm_C"]      = st.slider("C", 0.01, 20.0, 1.0, step=0.1, key="svm_c")
                params["svm_kernel"] = st.selectbox("Kernel", ["rbf", "linear", "poly"], key="svm_kernel")
            with c3:
                st.markdown("**Random Forest**")
                params["rf_n"]     = st.slider("Estimators", 10, 300, 100, step=10, key="rf_n_cls")
                params["rf_depth"] = st.slider("Max depth",   1,  30,  10,          key="rf_d_cls")
            with c4:
                st.markdown("**Decision Tree**")
                params["dt_depth"] = st.slider("Max depth", 1, 30, 5, key="dt_d")

        else:  # Regression
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**SVR**")
                params["svr_C"]      = st.slider("C",       0.01, 20.0, 1.0, step=0.1,  key="svr_c")
                params["svr_kernel"] = st.selectbox("Kernel", ["rbf", "linear", "poly"], key="svr_kernel")
                params["svr_eps"]    = st.slider("Epsilon", 0.01, 1.0,  0.1, step=0.01, key="svr_eps")
            with c2:
                st.markdown("**Random Forest**")
                params["rf_n"]     = st.slider("Estimators", 10, 300, 100, step=10, key="rf_n_reg")
                params["rf_depth"] = st.slider("Max depth",   1,  30,  10,          key="rf_d_reg")
            with c3:
                st.markdown("**Linear Regression**")
                st.info("No hyperparameters to tune for OLS Linear Regression.")

        st.markdown("---")

        # ── Train button ────────────────────────────────────────
        if st.button("🚀  TRAIN ALL MODELS"):
            with st.spinner("Training models…"):
                try:
                    X_tr, X_te, y_tr, y_te, scaler, encoders, le, feat_cols = \
                        models.preprocess_data(df, target, task)

                    st.session_state.update({
                        "X_train": X_tr, "X_test": X_te,
                        "y_train": y_tr, "y_test": y_te,
                        "scaler": scaler, "encoders": encoders,
                        "label_encoder": le, "feature_cols": feat_cols,
                    })

                    model_dict = (
                        models.get_classification_models(params)
                        if task == "Classification"
                        else models.get_regression_models(params)
                    )

                    trained, metrics = models.train_all_models(
                        model_dict, X_tr, y_tr, X_te, y_te, task
                    )
                    st.session_state["trained_models"] = trained
                    st.session_state["metrics"]        = metrics
                    st.success("✓ All models trained successfully!")

                except Exception as e:
                    st.error(f"Training error: {e}")

        # ── Metrics table & comparison chart ───────────────────
        if st.session_state["metrics"]:
            st.markdown("---")
            st.markdown('<p class="section-label">Performance Comparison</p>', unsafe_allow_html=True)

            metrics_df = (
                pd.DataFrame(st.session_state["metrics"])
                .T.reset_index()
                .rename(columns={"index": "Model"})
            )
            numeric_cols = [c for c in metrics_df.columns if c != "Model"]
            st.dataframe(
                metrics_df.style
                    .highlight_max(subset=numeric_cols, color="#1a3a1a")
                    .highlight_min(
                        subset=[c for c in numeric_cols if c not in ("R²", "Accuracy")],
                        color="#3a1a1a"
                    ),
                use_container_width=True
            )

            if task == "Classification":
                fig = models.plot_accuracy_comparison(st.session_state["metrics"])
            else:
                fig = models.plot_regression_metrics(st.session_state["metrics"])

            st.pyplot(fig, use_container_width=False)
            close_fig(fig)


# ═══════════════════════════════════════════════════════════════
# TAB 3 — EVALUATE
# ═══════════════════════════════════════════════════════════════
with tab3:
    if not st.session_state["trained_models"]:
        no_models_warning()
    else:
        trained = st.session_state["trained_models"]
        X_test  = st.session_state["X_test"]
        y_test  = st.session_state["y_test"]
        task    = st.session_state["task_type"]

        if task == "Classification":

            # Confusion matrices
            st.markdown('<p class="section-label">Confusion Matrices</p>', unsafe_allow_html=True)
            fig_cm = models.plot_confusion_matrices(trained, X_test, y_test)
            st.pyplot(fig_cm, use_container_width=True)
            close_fig(fig_cm)

            st.markdown("---")

            # ROC curves
            st.markdown('<p class="section-label">ROC Curves (Binary Classification)</p>', unsafe_allow_html=True)
            fig_roc = models.plot_roc_curves(trained, X_test, y_test)
            if fig_roc:
                st.pyplot(fig_roc, use_container_width=False)
                close_fig(fig_roc)
            else:
                st.info("ROC curves are only available for binary classification tasks.")

            st.markdown("---")

            # Per-model classification report
            st.markdown('<p class="section-label">Classification Report</p>', unsafe_allow_html=True)
            selected = st.selectbox("Select model", list(trained.keys()), key="eval_model")
            report_df = models.get_classification_report(trained[selected], X_test, y_test)
            st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

        else:  # Regression

            st.markdown('<p class="section-label">Actual vs Predicted</p>', unsafe_allow_html=True)
            fig_scatter = models.plot_regression_scatter(trained, X_test, y_test)
            st.pyplot(fig_scatter, use_container_width=True)
            close_fig(fig_scatter)

            st.markdown("---")

            st.markdown('<p class="section-label">Metric Comparison</p>', unsafe_allow_html=True)
            fig_metrics = models.plot_regression_metrics(st.session_state["metrics"])
            st.pyplot(fig_metrics, use_container_width=False)
            close_fig(fig_metrics)


# ═══════════════════════════════════════════════════════════════
# TAB 4 — PREDICT
# ═══════════════════════════════════════════════════════════════
with tab4:
    if not st.session_state["trained_models"]:
        no_models_warning()
    else:
        trained       = st.session_state["trained_models"]
        scaler        = st.session_state["scaler"]
        encoders      = st.session_state["encoders"]
        feature_cols  = st.session_state["feature_cols"]
        label_encoder = st.session_state["label_encoder"]
        task          = st.session_state["task_type"]

        c1, c2 = st.columns([1, 2])

        with c1:
            st.markdown('<p class="section-label">Select Model</p>', unsafe_allow_html=True)
            chosen = st.selectbox("Model", list(trained.keys()), key="pred_model")
            # Show that model's metrics as a badge
            if chosen in st.session_state["metrics"]:
                badge = " · ".join(
                    f"{k}: {v}"
                    for k, v in st.session_state["metrics"][chosen].items()
                )
                st.markdown(f'<span class="model-badge">{badge}</span>', unsafe_allow_html=True)

        with c2:
            st.markdown('<p class="section-label">Upload Prediction Data</p>', unsafe_allow_html=True)
            pred_file = st.file_uploader(
                "CSV with same features as training data (no target column required)",
                type=["csv"],
                key="pred_upload"
            )

        if pred_file:
            try:
                pred_df = pd.read_csv(pred_file)

                # Preprocess using saved artifacts (from models.py)
                pred_scaled = models.preprocess_prediction_input(
                    pred_df, feature_cols, encoders, scaler
                )

                predictions = trained[chosen].predict(pred_scaled)

                # Decode labels back to original strings if applicable
                if label_encoder is not None and task == "Classification":
                    predictions = label_encoder.inverse_transform(predictions)

                result_df = pred_df.copy()
                result_df["🔮 Prediction"] = predictions

                st.markdown("---")
                st.markdown('<p class="section-label">Prediction Results</p>', unsafe_allow_html=True)
                st.dataframe(result_df, use_container_width=True)

                csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️  Download Predictions as CSV",
                    data=csv_bytes,
                    file_name="predictions.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"Prediction error: {e}")

        else:
            st.markdown(
                '<div class="status-info">ℹ️  Upload a CSV with the same features as your '
                'training data.<br>The target column is NOT required. '
                'Predictions will be added as a new column.</div>',
                unsafe_allow_html=True
            )


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#2d3748; font-family:'Space Mono',monospace;
            font-size:0.7rem; letter-spacing:2px; padding:1rem 0;">
    ML STUDIO · BUILT WITH STREAMLIT + SCIKIT-LEARN
</div>
""", unsafe_allow_html=True)
