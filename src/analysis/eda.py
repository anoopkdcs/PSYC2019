# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 12:14:24 2026

@author: Anoop Kadan
"""
import os
import sys 
# eda.py -> analysis -> src -> project root
CURRENT_FILE = os.path.abspath(__file__)
ANALYSIS_DIR = os.path.dirname(CURRENT_FILE)
SRC_DIR = os.path.dirname(ANALYSIS_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.data_io import load_analysis_dataset
import numpy as np
from scipy.stats import pearsonr
from pandas.plotting import scatter_matrix
from statsmodels.stats.outliers_influence import variance_inflation_factor


def eda_distribution_and_quality_check(
    df: pd.DataFrame,
    output_folder: str
) -> None:
    """
    Perform distribution and data quality checks for the analysis-ready dataset.

    Saves:
    - combined_histograms.png
    - combined_boxplots.png
    - eda_report.txt
    """

    cols = [
        'Module mark',
        'Hours in Course',
        'Lecture attendance %',
        'Workshop attendance %',
        'Feedback attendance %',
        'Total attendance %'
    ]

    missing_cols = [col for col in cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    os.makedirs(output_folder, exist_ok=True)

    data = df[cols].copy()

    for col in cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # -----------------------------
    # 1. Summary statistics
    # -----------------------------
    summary_stats = data.describe().T

    # -----------------------------
    # 2. Missing values
    # -----------------------------
    missing_values = data.isna().sum()

    # -----------------------------
    # 3. Range checks
    # -----------------------------
    range_lines = []
    for col in cols:
        range_lines.append(
            f"{col}: min={data[col].min()}, max={data[col].max()}"
        )

    # -----------------------------
    # 4. Invalid attendance values
    # -----------------------------
    attendance_cols = [
        'Lecture attendance %',
        'Workshop attendance %',
        'Feedback attendance %',
        'Total attendance %'
    ]

    invalid_attendance_lines = []
    for col in attendance_cols:
        invalid = data[(data[col] < 0) | (data[col] > 100)][col]
        invalid_attendance_lines.append(
            f"{col}: {len(invalid)} invalid value(s)"
        )

    invalid_hours = data[data['Hours in Course'] < 0]['Hours in Course']
    invalid_hours_line = f"Hours in Course negative values: {len(invalid_hours)}"

    # -----------------------------
    # 5. IQR outlier check
    # -----------------------------
    outlier_summary = []

    for col in cols:
        series = data[col].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = series[(series < lower) | (series > upper)]

        outlier_summary.append({
            'Variable': col,
            'Lower bound': round(lower, 2),
            'Upper bound': round(upper, 2),
            'Number of outliers': int(outliers.shape[0])
        })

    outlier_df = pd.DataFrame(outlier_summary)

    # -----------------------------
    # 6. Save combined histograms
    # -----------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        axes[i].hist(data[col].dropna(), bins=15, edgecolor='black')
        axes[i].set_title(f'Histogram of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')

    plt.tight_layout()
    histogram_path = os.path.join(output_folder, "combined_histograms.png")
    plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # -----------------------------
    # 7. Save combined boxplots
    # -----------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        axes[i].boxplot(data[col].dropna(), vert=True)
        axes[i].set_title(f'Boxplot of {col}')
        axes[i].set_ylabel(col)

    plt.tight_layout()
    boxplot_path = os.path.join(output_folder, "combined_boxplots.png")
    plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # -----------------------------
    # 8. Save report to text file
    # -----------------------------
    report_path = os.path.join(output_folder, "eda_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("EDA DISTRIBUTION AND DATA QUALITY REPORT\n")
        f.write("=" * 50 + "\n\n")

        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 50 + "\n")
        f.write(summary_stats.to_string())
        f.write("\n\n")

        f.write("MISSING VALUES\n")
        f.write("-" * 50 + "\n")
        f.write(missing_values.to_string())
        f.write("\n\n")

        f.write("RANGE CHECKS\n")
        f.write("-" * 50 + "\n")
        for line in range_lines:
            f.write(line + "\n")
        f.write("\n")

        f.write("ATTENDANCE VALUES OUTSIDE 0-100\n")
        f.write("-" * 50 + "\n")
        for line in invalid_attendance_lines:
            f.write(line + "\n")
        f.write("\n")

        f.write("NEGATIVE HOURS CHECK\n")
        f.write("-" * 50 + "\n")
        f.write(invalid_hours_line + "\n\n")

        f.write("IQR OUTLIER CHECK\n")
        f.write("-" * 50 + "\n")
        f.write(outlier_df.to_string(index=False))
        f.write("\n")

    print(f"Saved histograms: {histogram_path}")
    print(f"Saved boxplots: {boxplot_path}")
    print(f"Saved report: {report_path}")


def eda_relationship_with_final_mark(
    df: pd.DataFrame,
    output_folder: str
) -> None:
    """
    Explore the relationship between engagement variables and final module mark.

    Saves:
    - engagement_vs_module_mark.png
    - engagement_vs_module_mark_correlations.csv
    - engagement_vs_module_mark_report.txt
    """

    target_col = 'Module mark'
    predictor_cols = [
        'Hours in Course',
        'Lecture attendance %',
        'Workshop attendance %',
        'Feedback attendance %',
        'Total attendance %'
    ]

    required_cols = [target_col] + predictor_cols
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    os.makedirs(output_folder, exist_ok=True)

    data = df[required_cols].copy()

    for col in required_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # -----------------------------------
    # 1. Scatterplots with fitted lines
    # -----------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    correlation_rows = []

    for i, predictor in enumerate(predictor_cols):
        ax = axes[i]

        plot_data = data[[predictor, target_col]].dropna()

        x = plot_data[predictor]
        y = plot_data[target_col]

        ax.scatter(x, y)

        # Fitted line
        if len(plot_data) >= 2:
            m, b = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = m * x_line + b
            ax.plot(x_line, y_line)

            r, p = pearsonr(x, y)
        else:
            r, p = np.nan, np.nan

        
        alpha = 0.05
        correlation_rows.append({
            'Predictor': predictor,
            'Pearson r': round(r, 4) if pd.notna(r) else np.nan,
            'p-value': f"{p:.2e}" if pd.notna(p) else np.nan,
            'Statistically significant': "Yes" if pd.notna(p) and p < alpha else "No",
            'N': len(plot_data)
        })

        ax.set_title(f'{predictor} vs {target_col}')
        ax.set_xlabel(predictor)
        ax.set_ylabel(target_col)

    # Remove unused last subplot
    if len(axes) > len(predictor_cols):
        fig.delaxes(axes[-1])

    plt.tight_layout()
    scatterplot_path = os.path.join(output_folder, "engagement_vs_module_mark.png")
    plt.savefig(scatterplot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # -----------------------------------
    # 2. Save correlation results
    # -----------------------------------
    correlation_df = pd.DataFrame(correlation_rows)

    correlation_csv_path = os.path.join(
        output_folder,
        "engagement_vs_module_mark_correlations.csv"
    )
    correlation_df.to_csv(correlation_csv_path, index=False)

    # -----------------------------------
    # 3. Save text report
    # -----------------------------------
    report_path = os.path.join(
        output_folder,
        "engagement_vs_module_mark_report.txt"
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("BIVARIATE RELATIONSHIPS WITH MODULE MARK\n")
        f.write("=" * 60 + "\n\n")
        f.write("This report shows Pearson correlations between each engagement\n")
        f.write("variable and final module mark.\n\n")
        f.write(correlation_df.to_string(index=False))
        f.write("\n")

    print(f"Saved scatterplots: {scatterplot_path}")
    print(f"Saved correlations CSV: {correlation_csv_path}")
    print(f"Saved report: {report_path}")




def eda_relationship_among_predictors(
    df: pd.DataFrame,
    output_folder: str
) -> None:
    """
    Explore relationships among predictor variables.

    Saves:
    - predictor_correlation_matrix.csv
    - predictor_correlation_heatmap.png
    - predictor_scatter_matrix.png
    - predictor_vif.csv
    - predictor_relationships_report.txt
    """

    predictor_cols = [
        'Hours in Course',
        'Lecture attendance %',
        'Workshop attendance %',
        'Feedback attendance %',
        'Total attendance %'
    ]

    missing_cols = [col for col in predictor_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    os.makedirs(output_folder, exist_ok=True)

    data = df[predictor_cols].copy()

    for col in predictor_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Keep rows with complete predictor data for correlation / VIF
    clean_data = data.dropna().copy()

    # -----------------------------------
    # 1. Correlation matrix
    # -----------------------------------
    corr_matrix = clean_data.corr(method='pearson')

    corr_csv_path = os.path.join(output_folder, "predictor_correlation_matrix.csv")
    corr_matrix.to_csv(corr_csv_path)

    # -----------------------------------
    # 2. Correlation heatmap
    # -----------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(corr_matrix, aspect='auto')
    fig.colorbar(cax)

    ax.set_xticks(range(len(predictor_cols)))
    ax.set_yticks(range(len(predictor_cols)))
    ax.set_xticklabels(predictor_cols, rotation=45, ha='left')
    ax.set_yticklabels(predictor_cols)

    for i in range(len(predictor_cols)):
        for j in range(len(predictor_cols)):
            ax.text(
                j, i,
                f"{corr_matrix.iloc[i, j]:.2f}",
                va='center',
                ha='center'
            )

    plt.title("Correlation Heatmap of Predictors", pad=20)
    plt.tight_layout()

    heatmap_path = os.path.join(output_folder, "predictor_correlation_heatmap.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # -----------------------------------
    # 3. Scatter matrix
    # -----------------------------------
    scatter_fig = plt.figure(figsize=(12, 12))
    scatter_matrix(
        clean_data,
        figsize=(12, 12),
        diagonal='hist',
        alpha=0.7
    )
    plt.suptitle("Scatter Matrix of Predictors", y=0.92)

    scatter_matrix_path = os.path.join(output_folder, "predictor_scatter_matrix.png")
    plt.savefig(scatter_matrix_path, dpi=300, bbox_inches='tight')
    plt.close()

    # -----------------------------------
    # 4. VIF calculation
    # -----------------------------------
    vif_data = clean_data.copy()

    vif_rows = []
    for i, col in enumerate(vif_data.columns):
        vif_value = variance_inflation_factor(vif_data.values, i)
        vif_rows.append({
            'Predictor': col,
            'VIF': round(vif_value, 4)
        })

    vif_df = pd.DataFrame(vif_rows)
    vif_csv_path = os.path.join(output_folder, "predictor_vif.csv")
    vif_df.to_csv(vif_csv_path, index=False)

    # -----------------------------------
    # 5. Save report
    # -----------------------------------
    report_path = os.path.join(output_folder, "predictor_relationships_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("RELATIONSHIPS AMONG PREDICTORS\n")
        f.write("=" * 60 + "\n\n")

        f.write("PREDICTOR VARIABLES\n")
        f.write("-" * 60 + "\n")
        for col in predictor_cols:
            f.write(f"{col}\n")
        f.write("\n")

        f.write("CORRELATION MATRIX\n")
        f.write("-" * 60 + "\n")
        f.write(corr_matrix.to_string())
        f.write("\n\n")

        f.write("VARIANCE INFLATION FACTOR (VIF)\n")
        f.write("-" * 60 + "\n")
        f.write(vif_df.to_string(index=False))
        f.write("\n\n")

        f.write("INTERPRETATION GUIDE\n")
        f.write("-" * 60 + "\n")
        f.write("High correlations among predictors may indicate redundancy.\n")
        f.write("VIF > 5 suggests possible multicollinearity.\n")
        f.write("VIF > 10 suggests serious multicollinearity.\n")
        f.write("Because 'Total attendance %' is derived from class-specific attendance,\n")
        f.write("it may be highly correlated with lecture, workshop, and feedback attendance.\n")

    print(f"Saved correlation matrix: {corr_csv_path}")
    print(f"Saved heatmap: {heatmap_path}")
    print(f"Saved scatter matrix: {scatter_matrix_path}")
    print(f"Saved VIF file: {vif_csv_path}")
    print(f"Saved report: {report_path}")


def main() -> None:
    analysis_df = load_analysis_dataset()

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    output_folder = os.path.join(project_root, "data", "eda_outputs")

    # eda_distribution_and_quality_check(analysis_df, output_folder)
    # eda_relationship_with_final_mark(analysis_df, output_folder)
    eda_relationship_among_predictors(analysis_df, output_folder)


if __name__ == "__main__":
    main()