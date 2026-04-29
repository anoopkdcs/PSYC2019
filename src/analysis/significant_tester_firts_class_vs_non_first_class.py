# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 20:58:07 2026

@author: ak7u24
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
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu
from src.utils.data_io import load_analysis_dataset


def check_first_class_attendance_by_type(
    df: pd.DataFrame,
    first_class_cutoff: float = 69 #70.0
) -> pd.DataFrame:
    """
    Compare first-class vs non-first-class students on:
    - Lecture attendance %
    - Workshop attendance %
    - Feedback attendance %

    First class is defined as Module mark >= first_class_cutoff.
    """
    attendance_cols = [
        'Lecture attendance %',
        'Workshop attendance %',
        'Feedback attendance %'
    ]

    required_cols = ['Module mark'] + attendance_cols
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    data = df[required_cols].copy()

    data['Module mark'] = pd.to_numeric(data['Module mark'], errors='coerce')
    for col in attendance_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    data = data.dropna(subset=['Module mark'])
    data['First class'] = data['Module mark'] >= first_class_cutoff

    results = []

    for col in attendance_cols:
        temp = data[['First class', col]].dropna()

        first_group = temp.loc[temp['First class'], col]
        non_first_group = temp.loc[~temp['First class'], col]

        t_stat, t_p = ttest_ind(
            first_group,
            non_first_group,
            equal_var=False,
            nan_policy='omit'
        )

        u_stat, u_p = mannwhitneyu(
            first_group,
            non_first_group,
            alternative='two-sided'
        )

        mean_first = first_group.mean()
        mean_non_first = non_first_group.mean()

        if mean_first > mean_non_first:
            direction = "First-class higher"
        elif mean_first < mean_non_first:
            direction = "Non-first-class higher"
        else:
            direction = "Equal means"

        results.append({
            'Attendance type': col,
            'N first-class': len(first_group),
            'N non-first-class': len(non_first_group),
            'Mean first-class': round(mean_first, 2),
            'Mean non-first-class': round(mean_non_first, 2),
            'Median first-class': round(first_group.median(), 2),
            'Median non-first-class': round(non_first_group.median(), 2),
            'Welch t': round(t_stat, 4),
            'Welch p': t_p,
            'Welch p (exp)': f"{t_p:.2e}",
            'Welch significant': 'Yes' if t_p < 0.05 else 'No',
            'Mann-Whitney U': round(u_stat, 4),
            'MWU p': u_p,
            'MWU p (exp)': f"{u_p:.2e}",
            'MWU significant': 'Yes' if u_p < 0.05 else 'No',
            'Direction': direction
        })

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    return results_df, data 


def save_first_class_attendance_report(
    results_df: pd.DataFrame,
    filename: str = "first_class_attendance_report.txt"
) -> str:
    """
    Save the attendance comparison report to:
    project_root/data/report_outputs/<filename>
    """
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    output_folder = os.path.join(project_root, "data", "sig_test_outputs")
    os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(output_folder, filename)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("FIRST-CLASS VS NON-FIRST-CLASS ATTENDANCE REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write("First class defined as: Module mark >= 69\n\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n")

        f.write("INTERPRETATION\n")
        f.write("-" * 60 + "\n")
        for _, row in results_df.iterrows():
            f.write(f"{row['Attendance type']}:\n")
            f.write(
                f"  Mean attendance was {row['Mean first-class']}% for first-class students "
                f"and {row['Mean non-first-class']}% for non-first-class students.\n"
            )
            f.write(
                f"  Welch t-test: t = {row['Welch t']}, p = {row['Welch p (exp)']} "
                f"({row['Welch significant']}).\n"
            )
            f.write(
                f"  Mann-Whitney U test: U = {row['Mann-Whitney U']}, "
                f"p = {row['MWU p (exp)']} ({row['MWU significant']}).\n"
            )
            f.write(f"  Direction: {row['Direction']}.\n\n")

    return output_path

def save_first_class_attendance_boxplot(
    df: pd.DataFrame,
    first_class_cutoff: float = 69.0,
    filename: str = "first_class_attendance_boxplot.png"
) -> str:
    """
    Save boxplots for:
    - Lecture attendance %
    - Workshop attendance %
    - Feedback attendance %
    """
    attendance_cols = [
        'Lecture attendance %',
        'Workshop attendance %',
        'Feedback attendance %'
    ]

    required_cols = ['Module mark'] + attendance_cols
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    data = df[required_cols].copy()
    data['Module mark'] = pd.to_numeric(data['Module mark'], errors='coerce')

    for col in attendance_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    data = data.dropna(subset=['Module mark'])
    data['First class'] = data['Module mark'] >= first_class_cutoff

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    output_folder = os.path.join(project_root, "data", "sig_test_outputs")
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, filename)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, col in zip(axes, attendance_cols):
        temp = data[['First class', col]].dropna()

        first_group = temp.loc[temp['First class'], col]
        non_first_group = temp.loc[~temp['First class'], col]

        ax.boxplot(
            [first_group, non_first_group],
            tick_labels=['First class', 'Non-first class']
        )
        ax.set_title(col)
        ax.set_ylabel('Attendance %')

    plt.suptitle('Attendance by first-class status', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    return output_path

def main() -> None:
    analysis_df = load_analysis_dataset()

    results_df, data = check_first_class_attendance_by_type(analysis_df)
    report_path = save_first_class_attendance_report(results_df)
    boxplot_path = save_first_class_attendance_boxplot(analysis_df)

    print(f"Saved report: {report_path}")
    print(f"Saved box plot: {boxplot_path}")


if __name__ == "__main__":
    main()
