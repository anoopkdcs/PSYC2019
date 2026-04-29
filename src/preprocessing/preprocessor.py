# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 16:31:28 2026

@author: ak7u24
"""

import pandas as pd
import os
from pathlib import Path
import json

def read_excel_file(filename):
    """
    Reads an Excel file from the 'data' folder.

    Parameters:
        filename (str): The name of the Excel file to read.

    Returns:
        DataFrame: A pandas DataFrame containing the data from the Excel file.
    """
    folder_path = os.path.join(os.path.dirname(__file__), 'data')
    file_path = os.path.join(folder_path, filename)
    
    try:
        data = pd.read_excel(file_path)
        return data
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found in the 'data' folder.")
    except Exception as e:
        print(f"An error occurred: {e}")


def combine_marks_and_bb_hours(marks: pd.DataFrame, bb_cource_hour: pd.DataFrame) -> pd.DataFrame:
    """
    Combine marks and Blackboard course hours into a single DataFrame.

    Parameters
    ----------
    marks : pd.DataFrame
        DataFrame with columns:
        ['Student ID', 'Module mark']

    bb_cource_hour : pd.DataFrame
        DataFrame with columns:
        ['Surname', 'First Name', 'Student ID', 'Hours in Course']

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        ['Student ID', 'Surname', 'First Name', 'Module mark', 'Hours in Course']
    """
    marks_clean = marks.copy()
    bb_hours_clean = bb_cource_hour.copy()

    # Convert Student ID to integer
    marks_clean['Student ID'] = pd.to_numeric(marks_clean['Student ID'], errors='coerce').astype('Int64')
    bb_hours_clean['Student ID'] = pd.to_numeric(bb_hours_clean['Student ID'], errors='coerce').astype('Int64')

    # Round Hours in Course to 2 decimal places
    bb_hours_clean['Hours in Course'] = pd.to_numeric(
        bb_hours_clean['Hours in Course'], errors='coerce'
    ).round(2)

    combined_data = pd.merge(
        bb_hours_clean[['Student ID', 'Surname', 'First Name', 'Hours in Course']],
        marks_clean[['Student ID', 'Module mark']],
        on='Student ID',
        how='inner'
    )

    combined_data = combined_data[
        ['Student ID', 'Surname', 'First Name', 'Module mark', 'Hours in Course']
    ]

    return combined_data


def check_student_by_id(data: pd.DataFrame, student_id: int) -> None:
    """
    Print the row(s) for a specific student ID.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing a 'Student ID' column.
    student_id : int
        The student ID to search for.
    """
    student = data[data['Student ID'] == student_id]

    if student.empty:
        print(f"Student ID {student_id} not found.")
    else:
        print(student.to_string(index=False))


def update_features_with_attendance(
    features: pd.DataFrame,
    timetable: pd.DataFrame,
    day_level_attendance: pd.DataFrame,
    max_classes: int = 12
) -> pd.DataFrame:
    """
    Update `features` with lecture/workshop/feedback attendance counts and percentages.

    Parameters
    ----------
    features : pd.DataFrame
        Columns:
        ['Student ID', 'Surname', 'First Name', 'Module mark', 'Hours in Course']

    timetable : pd.DataFrame
        Columns:
        ['week', 'Lecture', 'workshop', 'feedback']

    day_level_attendance : pd.DataFrame
        Columns:
        ['Type', 'Student No', 'First Name', 'Surname', 'Date', 'Lesson Type', '% Attended']

    max_classes : int, default=12
        Maximum number of timetable dates to consider for each class type.

    Returns
    -------
    pd.DataFrame
        Updated features DataFrame with new columns:
        - Lecture attended
        - Lecture attendance %
        - Workshop attended
        - Workshop attendance %
        - Feedback attended
        - Feedback attendance %
    """

    feat = features.copy()
    tt = timetable.copy()
    att = day_level_attendance.copy()

    # -------------------------
    # 1. Clean IDs
    # -------------------------
    feat['Student ID'] = pd.to_numeric(feat['Student ID'], errors='coerce').astype('Int64')
    att['Student No'] = pd.to_numeric(att['Student No'], errors='coerce').astype('Int64')

    # -------------------------
    # 2. Clean dates
    # -------------------------
    for col in ['Lecture', 'workshop', 'feedback']:
        tt[col] = pd.to_datetime(tt[col], errors='coerce', dayfirst=True).dt.normalize()

    att['Date'] = pd.to_datetime(att['Date'], errors='coerce', dayfirst=True).dt.normalize()

    # -------------------------
    # 3. Build timetable date sets
    # -------------------------
    def get_class_dates(column_name: str) -> set:
        dates = sorted(tt[column_name].dropna().drop_duplicates())
        return dates 

    lecture_dates = get_class_dates('Lecture')
    workshop_dates = get_class_dates('workshop')
    feedback_dates = get_class_dates('feedback')

    lecture_total = max_classes
    workshop_total = max_classes
    feedback_total = max_classes

    # -------------------------
    # 4. Keep only present records
    # -------------------------
    att['Type'] = att['Type'].astype(str).str.strip().str.upper()

    present = att.loc[
        att['Type'] == 'IN',
        ['Student No', 'Date']
    ].dropna(subset=['Student No', 'Date']).drop_duplicates()

    # -------------------------
    # 5. Count attendance by class type
    # -------------------------
    lecture_counts = (
        present[present['Date'].isin(lecture_dates)]
        .groupby('Student No')['Date']
        .nunique()
        .rename('Lecture attended')
    )

    workshop_counts = (
        present[present['Date'].isin(workshop_dates)]
        .groupby('Student No')['Date']
        .nunique()
        .rename('Workshop attended')
    )

    feedback_counts = (
        present[present['Date'].isin(feedback_dates)]
        .groupby('Student No')['Date']
        .nunique()
        .rename('Feedback attended')
    )

    attendance_summary = pd.concat(
        [lecture_counts, workshop_counts, feedback_counts],
        axis=1
    ).fillna(0).reset_index()

    attendance_summary = attendance_summary.rename(columns={'Student No': 'Student ID'})

    # Convert counts to integers
    for col in ['Lecture attended', 'Workshop attended', 'Feedback attended']:
        attendance_summary[col] = attendance_summary[col].astype(int)

    # -------------------------
    # 6. Merge into features
    # -------------------------
    updated_features = feat.merge(attendance_summary, on='Student ID', how='left')

    for col in ['Lecture attended', 'Workshop attended', 'Feedback attended']:
        updated_features[col] = updated_features[col].fillna(0).astype(int)

    # -------------------------
    # 7. Calculate percentages
    # -------------------------
    updated_features['Lecture attendance %'] = (
        (updated_features['Lecture attended'] / lecture_total * 100).round(2)
        if lecture_total > 0 else 0.0
    )

    updated_features['Workshop attendance %'] = (
        (updated_features['Workshop attended'] / workshop_total * 100).round(2)
        if workshop_total > 0 else 0.0
    )

    updated_features['Feedback attendance %'] = (
        (updated_features['Feedback attended'] / feedback_total * 100).round(2)
        if feedback_total > 0 else 0.0
    )

    # -------------------------
    # 8. Reorder columns
    # -------------------------
    updated_features = updated_features[
        [
            'Student ID',
            'Surname',
            'First Name',
            'Module mark',
            'Hours in Course',
            'Lecture attended',
            'Lecture attendance %',
            'Workshop attended',
            'Workshop attendance %',
            'Feedback attended',
            'Feedback attendance %'
        ]
    ]

    return updated_features


def add_total_attendance_seats(
    features: pd.DataFrame,
    total_attendance_seats: pd.DataFrame
) -> pd.DataFrame:
    """
    Add overall SEAtS attendance percentage to the end of the features DataFrame.

    Parameters
    ----------
    features : pd.DataFrame
        DataFrame containing a 'Student ID' column.

    total_attendance_seats : pd.DataFrame
        DataFrame with columns:
        ['Student No', 'Student Name', '% Attended']

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with a new column:
        ['SEAtS overall attendance %']
    """
    feat = features.copy()
    seats = total_attendance_seats.copy()

    # Clean ID columns
    feat['Student ID'] = pd.to_numeric(feat['Student ID'], errors='coerce').astype('Int64')
    seats['Student No'] = pd.to_numeric(seats['Student No'], errors='coerce').astype('Int64')

    # Clean attendance column
    seats['% Attended'] = pd.to_numeric(seats['% Attended'], errors='coerce').round(2)

    # Keep only needed columns and rename for merge
    seats = seats[['Student No', '% Attended']].rename(
        columns={
            'Student No': 'Student ID',
            '% Attended': 'SEAtS overall attendance %'
        }
    )

    # Merge
    updated_features = feat.merge(seats, on='Student ID', how='left')

    # Move the new column to the end
    cols = [col for col in updated_features.columns if col != 'SEAtS overall attendance %']
    cols.append('SEAtS overall attendance %')
    updated_features = updated_features[cols]

    return updated_features


def save_final_features(
    features: pd.DataFrame,
    filename: str = "features_final",
    subfolder: str = "processed_data"
) -> None:
    """
    Save the final features DataFrame inside:
    parent_of_script/data/<subfolder>/

    Files saved
    -----------
    - features_final.parquet
    - features_final.csv
    - features_final_schema.json
    """
    df = features.copy()

    numeric_cols = [
        'Student ID',
        'Module mark',
        'Hours in Course',
        'Lecture attended',
        'Lecture attendance %',
        'Workshop attended',
        'Workshop attendance %',
        'Feedback attended',
        'Feedback attendance %',
        'Total attended',
        'Total attendance %',
        'SEAtS overall attendance %'
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'Student ID' in df.columns:
        df['Student ID'] = df['Student ID'].astype('Int64')

    # Go to the parent folder of the python file
    parent_folder = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    print(parent_folder)

    # Existing data folder in that parent folder
    data_folder = os.path.join(parent_folder, 'data')

    # New folder inside data
    output_folder = os.path.join(data_folder, subfolder)
    os.makedirs(output_folder, exist_ok=True)

    parquet_path = os.path.join(output_folder, f"{filename}.parquet")
    csv_path = os.path.join(output_folder, f"{filename}.csv")
    schema_path = os.path.join(output_folder, f"{filename}_schema.json")

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False, encoding="utf-8")

    schema = {
        "rows": int(df.shape[0]),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }

    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=4)

    # print(f"Saved: {parquet_path}")
    # print(f"Saved: {csv_path}")
    # print(f"Saved: {schema_path}")




if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print("Reading Excel file...")
    marks = read_excel_file(os.path.join(root_dir, 'data', 'marks.xlsx'))
    bb_hours = read_excel_file(os.path.join(root_dir, 'data', 'bb_overall_hours_in_module.xlsx')) 
    
    features = combine_marks_and_bb_hours(marks, bb_hours)
    timetable = read_excel_file(os.path.join(root_dir, 'data', 'timetable.xlsx'))
    day_level_attendance = read_excel_file(os.path.join(root_dir, 'data', 'seats_day_level.xlsx'))
    features = update_features_with_attendance(features, timetable, day_level_attendance)
    
    features['Total attended'] = (features['Lecture attended'] +
    features['Workshop attended'] + features['Feedback attended'])
    features['Total attendance %'] = (features['Total attended'] / 36 * 100).round(2) #12 (LEC) + 12 (WS)  + 12 (FB) = 36
    
    total_attendance_seats =  read_excel_file(os.path.join(root_dir, 'data', 'seats_overall.xlsx'))
    features = add_total_attendance_seats(features, total_attendance_seats)
    
    save_final_features(features)
    
    print(check_student_by_id(features, 36651311))
    


