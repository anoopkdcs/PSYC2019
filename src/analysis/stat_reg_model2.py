# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 16:19:36 2026

@author: ak7u24

Use one class-specific model:
Hours in Course
Lecture attendance %
Workshop attendance %
Feedback attendance %

"""
import os
import sys 
# model_regrssion.py -> analysis -> src -> project root
CURRENT_FILE = os.path.abspath(__file__)
ANALYSIS_DIR = os.path.dirname(CURRENT_FILE)
SRC_DIR = os.path.dirname(ANALYSIS_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from src.utils.data_io import load_analysis_dataset


def fit_class_specific_attendance_model(df: pd.DataFrame):
    """
    Fit the regression model:
    Module mark ~ Hours in Course
                 + Lecture attendance %
                 + Workshop attendance %
                 + Feedback attendance %
    """
    required_cols = [
        'Module mark',
        'Hours in Course',
        'Lecture attendance %',
        'Workshop attendance %',
        'Feedback attendance %'
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    data = df[required_cols].copy()

    for col in required_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    data = data.dropna()

    X = data[
        [
            'Hours in Course',
            'Lecture attendance %',
            'Workshop attendance %',
            'Feedback attendance %'
        ]
    ]
    y = data['Module mark']

    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    return model, data


def save_model_summary(
    model,
    filename: str = "reg_model2_summary.txt"
) -> str:
    """
    Save the regression summary to:
    project_root/data/model_outputs/<filename>
    """
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    output_folder = os.path.join(project_root, "data", "model_outputs")
    os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(output_folder, filename)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(model.summary().as_text())

    return output_path


def class_specific_coefficient_plot(
    model,
    filename: str = "reg_model2_coefficient_plot.png"
) -> str:
    """
    Create and save a coefficient plot with 95% confidence intervals.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    output_folder = os.path.join(project_root, "data", "model_outputs")
    os.makedirs(output_folder, exist_ok=True)

    coef_df = pd.DataFrame({
        'Predictor': model.params.index,
        'Coefficient': model.params.values
    })

    conf_int = model.conf_int()
    coef_df['CI Lower'] = conf_int[0].values
    coef_df['CI Upper'] = conf_int[1].values

    # Remove intercept
    coef_df = coef_df[coef_df['Predictor'] != 'const'].reset_index(drop=True)

    y_pos = np.arange(len(coef_df))
    x = coef_df['Coefficient'].values
    xerr = np.vstack([
        x - coef_df['CI Lower'].values,
        coef_df['CI Upper'].values - x
    ])

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        x=x,
        y=y_pos,
        xerr=xerr,
        fmt='o',
        capsize=5
    )
    plt.axvline(0, linestyle='--')
    plt.yticks(y_pos, coef_df['Predictor'])
    plt.xlabel('Regression coefficient')
    plt.ylabel('Predictor')
    plt.title('Class-Specific Model: Coefficients with 95% CIs')
    plt.tight_layout()

    save_path = os.path.join(output_folder, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    return save_path


def class_specific_scatterplots(
    data: pd.DataFrame,
    filename: str = "reg_model2_scatterplots.png"
) -> str:
    """
    Create and save scatterplots with fitted lines for:
    - Hours in Course vs Module mark
    - Lecture attendance % vs Module mark
    - Workshop attendance % vs Module mark
    - Feedback attendance % vs Module mark
    """
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    output_folder = os.path.join(project_root, "data", "model_outputs")
    os.makedirs(output_folder, exist_ok=True)

    predictors = [
        'Hours in Course',
        'Lecture attendance %',
        'Workshop attendance %',
        'Feedback attendance %'
    ]
    target = 'Module mark'

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, predictor in zip(axes, predictors):
        plot_data = data[[predictor, target]].dropna()

        x = plot_data[predictor].values
        y = plot_data[target].values

        ax.scatter(x, y)

        if len(plot_data) >= 2:
            m, b = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = m * x_line + b
            ax.plot(x_line, y_line)

        ax.set_xlabel(predictor)
        ax.set_ylabel(target)
        ax.set_title(f'{predictor} vs {target}')

    plt.suptitle('Class-Specific Engagement vs Module Mark', y=1.02)
    plt.tight_layout()

    save_path = os.path.join(output_folder, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    return save_path


def main() -> None:
    analysis_df = load_analysis_dataset()

    model, model_data = fit_class_specific_attendance_model(analysis_df)

    print(model.summary())

    summary_path = save_model_summary(model)
    coef_plot_path = class_specific_coefficient_plot(model)
    scatter_plot_path = class_specific_scatterplots(model_data)

    print(f"Saved model summary: {summary_path}")
    print(f"Saved coefficient plot: {coef_plot_path}")
    print(f"Saved scatter plot: {scatter_plot_path}")


if __name__ == "__main__":
    main()