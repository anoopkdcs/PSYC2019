# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 16:31:28 2026

@author: ak7u24
"""

import os
import pandas as pd


def load_master_dataset(
    filename: str = "student_dataset.parquet",
    subfolder: str = "student_dataset"
) -> pd.DataFrame:
    """
    Load the final features dataset from the data folder.

    Parameters
    ----------
    filename : str
        Name of the file to load, e.g. 'student_dataset.parquet'
        or 'student_dataset.csv'.

    subfolder : str
        Folder inside /data where the file is stored.

    Returns
    -------
    pd.DataFrame
        Loaded features DataFrame.
    """
    # Go from this file: src/utils/data_io.py -> project_root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    print(project_root)
    data_folder = os.path.join(project_root, "data", subfolder)
    file_path = os.path.join(data_folder, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if filename.endswith(".parquet"):
        return pd.read_parquet(file_path)
    elif filename.endswith(".csv"):
        return pd.read_csv(file_path)
    else:
        raise ValueError("Supported file types are .parquet and .csv only")
           

def create_analysis_ready_dataset(master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the master dataset into the analysis-ready dataset.

    Final variables:
    - Student ID
    - Module mark
    - Hours in Course
    - Lecture attendance %
    - Workshop attendance %
    - Feedback attendance %
    - Total attendance %
    """
    required_columns = [
        'Student ID',
        'Module mark',
        'Hours in Course',
        'Lecture attendance %',
        'Workshop attendance %',
        'Feedback attendance %',
        'Total attendance %'
    ]

    missing_columns = [col for col in required_columns if col not in master_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    analysis_df = master_df[required_columns].copy()

    # Clean data types
    analysis_df['Student ID'] = pd.to_numeric(
        analysis_df['Student ID'], errors='coerce'
    ).astype('Int64')

    numeric_cols = [
        'Module mark',
        'Hours in Course',
        'Lecture attendance %',
        'Workshop attendance %',
        'Feedback attendance %',
        'Total attendance %'
    ]

    for col in numeric_cols:
        analysis_df[col] = pd.to_numeric(analysis_df[col], errors='coerce')

    # Optional: remove duplicate student IDs if any
    #analysis_df = analysis_df.drop_duplicates(subset=['Student ID'])

    # Optional: drop rows with missing outcome
    #analysis_df = analysis_df.dropna(subset=['Module mark'])

    # Optional: sort by Student ID
    #analysis_df = analysis_df.sort_values('Student ID').reset_index(drop=True)

    return analysis_df


def load_analysis_dataset(
    filename: str = "student_dataset.parquet",
    subfolder: str = "student_dataset"
) -> pd.DataFrame:
    """
    Load the master dataset and return the analysis-ready dataset.
    """
    master_df = load_master_dataset(filename=filename, subfolder=subfolder)
    return create_analysis_ready_dataset(master_df)