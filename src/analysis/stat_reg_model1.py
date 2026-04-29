# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 13:50:52 2026

@author: ak7u24

Use one overall attendance model: MODEL 1
Hours in Course
Total attendance %


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
    
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from src.utils.data_io import load_analysis_dataset


def fit_overall_attendance_model(df: pd.DataFrame):
    """
    Fit:
        Module mark ~ Hours in Course + Total attendance %
    """
    required_cols = ['Module mark', 'Hours in Course', 'Total attendance %']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    data = df[required_cols].copy()

    for col in required_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    data = data.dropna()

    X = data[['Hours in Course', 'Total attendance %']]
    y = data['Module mark']

    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    return model, data


def coefficient_plot(model, output_folder: str) -> None:
    """
    Option A:
    Coefficient plot with 95% confidence intervals.
    """
    os.makedirs(output_folder, exist_ok=True)

    coef_df = pd.DataFrame({
        'Predictor': model.params.index,
        'Coefficient': model.params.values
    })

    conf_int = model.conf_int()
    coef_df['CI Lower'] = conf_int[0].values
    coef_df['CI Upper'] = conf_int[1].values

    # Remove intercept for cleaner poster plot
    coef_df = coef_df[coef_df['Predictor'] != 'const'].reset_index(drop=True)

    y_pos = np.arange(len(coef_df))
    x = coef_df['Coefficient'].values
    xerr = np.vstack([
        x - coef_df['CI Lower'].values,
        coef_df['CI Upper'].values - x
    ])

    plt.figure(figsize=(8, 4.5))
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
    plt.title('Coefficient Plot with 95% Confidence Intervals')
    plt.tight_layout()

    save_path = os.path.join(output_folder, 'reg_molde1_coefficient_plot.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    print(f"Saved: {save_path}")


def scatterplots(data: pd.DataFrame, output_folder: str) -> None:
    """
    Option B:
    Two-panel scatterplot with fitted regression lines:
    - Hours in Course vs Module mark
    - Total attendance % vs Module mark
    """
    os.makedirs(output_folder, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    predictors = ['Hours in Course', 'Total attendance %']
    target = 'Module mark'

    for ax, predictor in zip(axes, predictors):
        plot_data = data[[predictor, target]].dropna()

        x = plot_data[predictor].values
        y = plot_data[target].values

        ax.scatter(x, y)

        # fitted line
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = m * x_line + b
        ax.plot(x_line, y_line)

        ax.set_xlabel(predictor)
        ax.set_ylabel(target)
        ax.set_title(f'{predictor} vs {target}')

    plt.suptitle('Engagement vs Module Mark', y=1.02)
    plt.tight_layout()

    save_path = os.path.join(output_folder, 'reg_molde1_scatterplots.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    print(f"Saved: {save_path}")

def save_model_summary(model, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(model.summary().as_text())
        
        
def main() -> None:
    analysis_df = load_analysis_dataset()
    model, model_data = fit_overall_attendance_model(analysis_df)
    

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    output_folder = os.path.join(project_root, 'data', 'model_outputs')
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "reg_molde1_summary.txt")

    # Option A
    coefficient_plot(model, output_folder)

    # Option B
    scatterplots(model_data, output_folder)
    
    # Save model summary 
    save_model_summary(model, output_path)



if __name__ == "__main__":
    main()