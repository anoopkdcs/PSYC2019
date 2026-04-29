# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 16:34:40 2026

@author: ak7u24

Set SYS path 
import os
import sys 
# model.py -> analysis -> src -> project root
CURRENT_FILE = os.path.abspath(__file__)
ANALYSIS_DIR = os.path.dirname(CURRENT_FILE)
SRC_DIR = os.path.dirname(ANALYSIS_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
"""
import os
import sys 
# model.py -> analysis -> src -> project root
CURRENT_FILE = os.path.abspath(__file__)
ANALYSIS_DIR = os.path.dirname(CURRENT_FILE)
SRC_DIR = os.path.dirname(ANALYSIS_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
    
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.utils.data_io import load_analysis_dataset


try:
    from xgboost import XGBClassifier
except ImportError as e:
    raise ImportError(
        "xgboost is required for this script. Install it with: pip install xgboost"
    ) from e


def load_binary_classification_dataset(first_class_cutoff: float = 69.0) -> pd.DataFrame:
    """
    Read the analysis dataset, select the ML variables, and create a binary target.

    Target:
    - 1 = first class (Module mark >= first_class_cutoff)
    - 0 = non-first class

    Returns
    -------
    pd.DataFrame
        Columns:
        - Hours in Course
        - Lecture attendance %
        - Workshop attendance %
        - Feedback attendance %
        - First class
    """
    df = load_analysis_dataset().copy()

    required_cols = [
        "Module mark",
        "Hours in Course",
        "Lecture attendance %",
        "Workshop attendance %",
        "Feedback attendance %",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    binary_df = df[required_cols].copy()

    for col in required_cols:
        binary_df[col] = pd.to_numeric(binary_df[col], errors="coerce")

    binary_df["First class"] = (binary_df["Module mark"] >= first_class_cutoff).astype(int)

    binary_df = binary_df.drop(columns=["Module mark"])
    binary_df = binary_df.dropna().reset_index(drop=True)

    return binary_df


def compare_classification_models(
    binary_df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
    output_filename: str = "classification_model_comparison.csv",
):
    """
    Develop and compare:
    - Logistic Regression
    - Random Forest
    - XGBoost

    Validation strategy:
    - Stratified 5-fold CV is used because it preserves the first-class/non-first-class
      proportion in each fold and gives a more stable estimate than LOOCV for a
      moderate-sized binary classification dataset.

    Metrics:
    - ROC-AUC
    - Accuracy
    - Precision
    - Recall
    - F1-score

    Returns
    -------
    results_df : pd.DataFrame
        Cross-validated model comparison table.
    fitted_models : dict
        All models fitted on the full dataset after CV.
    X : pd.DataFrame
        Predictor matrix.
    y : pd.Series
        Binary target.
    """
    required_cols = [
        "Hours in Course",
        "Lecture attendance %",
        "Workshop attendance %",
        "Feedback attendance %",
        "First class",
    ]
    missing_cols = [col for col in required_cols if col not in binary_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    X = binary_df[
        [
            "Hours in Course",
            "Lecture attendance %",
            "Workshop attendance %",
            "Feedback attendance %",
        ]
    ].copy()
    y = binary_df["First class"].copy()

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scoring = {
        "roc_auc": "roc_auc",
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
    }

    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=random_state)),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=None,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=300,
                        learning_rate=0.05,
                        max_depth=3,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }

    results = []
    fitted_models = {}

    for model_name, pipeline in models.items():
        cv_results = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )

        results.append(
            {
                "Model": model_name,
                "ROC-AUC (mean)": cv_results["test_roc_auc"].mean(),
                "ROC-AUC (std)": cv_results["test_roc_auc"].std(),
                "Accuracy (mean)": cv_results["test_accuracy"].mean(),
                "Accuracy (std)": cv_results["test_accuracy"].std(),
                "Precision (mean)": cv_results["test_precision"].mean(),
                "Recall (mean)": cv_results["test_recall"].mean(),
                "F1-score (mean)": cv_results["test_f1"].mean(),
            }
        )

        pipeline.fit(X, y)
        fitted_models[model_name] = pipeline

    results_df = pd.DataFrame(results).sort_values(
        by="ROC-AUC (mean)", ascending=False
    ).reset_index(drop=True)

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    output_folder = os.path.join(project_root, "data", "ml_outputs")
    os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(output_folder, output_filename)
    results_df.to_csv(output_path, index=False)

    print("Validation strategy: Stratified 5-fold cross-validation")
    print(
        "Why this is appropriate: it preserves class balance across folds and provides\n"
        "a more stable performance estimate than LOOCV for a moderate-sized binary dataset."
    )
    print(f"\nSaved model comparison to: {output_path}")

    return results_df, fitted_models, X, y


def plot_shap_summary(
    fitted_pipeline,
    X: pd.DataFrame,
    model_name: str,
    beeswarm_filename: str = "shap_beeswarm.png",
    bar_filename: str = "shap_bar.png",
):
    """
    Generate:
    - SHAP beeswarm plot
    - SHAP bar plot

    Notes
    -----
    - For Logistic Regression, scaled data are passed to SHAP because the model was trained on scaled values.
    - For Random Forest and XGBoost, imputed data are passed directly.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    output_folder = os.path.join(project_root, "data", "ml_outputs")
    os.makedirs(output_folder, exist_ok=True)

    imputer = fitted_pipeline.named_steps["imputer"]
    X_imputed = pd.DataFrame(
        imputer.transform(X),
        columns=X.columns,
        index=X.index,
    )

    if "scaler" in fitted_pipeline.named_steps:
        scaler = fitted_pipeline.named_steps["scaler"]
        X_for_shap = pd.DataFrame(
            scaler.transform(X_imputed),
            columns=X.columns,
            index=X.index,
        )
    else:
        X_for_shap = X_imputed

    model = fitted_pipeline.named_steps["model"]

    explainer = shap.Explainer(model, X_for_shap)
    shap_values = explainer(X_for_shap)

    beeswarm_path = os.path.join(output_folder, beeswarm_filename)
    plt.figure()
    shap.plots.beeswarm(shap_values, show=False)
    plt.title(f"SHAP Beeswarm Plot: {model_name}")
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=300, bbox_inches="tight")
    plt.close()

    bar_path = os.path.join(output_folder, bar_filename)
    plt.figure()
    shap.plots.bar(shap_values, show=False)
    plt.title(f"SHAP Bar Plot: {model_name}")
    plt.tight_layout()
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved SHAP beeswarm plot: {beeswarm_path}")
    print(f"Saved SHAP bar plot: {bar_path}")


def main() -> None:
    # Step 1: prepare binary ML dataset
    binary_df = load_binary_classification_dataset(first_class_cutoff=69.0)

    # Step 2: compare ML models
    results_df, fitted_models, X, y = compare_classification_models(binary_df)

    print("\nModel comparison")
    print(results_df.to_string(index=False))

    # Step 3: use the best ROC-AUC model for SHAP
    best_model_name = results_df.iloc[0]["Model"]
    best_model = fitted_models[best_model_name]

    print(f"\nBest model for SHAP: {best_model_name}")

    plot_shap_summary(
        fitted_pipeline=best_model,
        X=X,
        model_name=best_model_name,
        beeswarm_filename=f"{best_model_name.lower().replace(' ', '_')}_shap_beeswarm.png",
        bar_filename=f"{best_model_name.lower().replace(' ', '_')}_shap_bar.png",
    )


if __name__ == "__main__":
    main()






