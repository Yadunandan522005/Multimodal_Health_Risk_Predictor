from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# DATASET PATH
# ============================================================

DATA_PATH = (
    ROOT
    / "data"
    / "DATASETS_final_4_diseases.xlsx"
)


# ============================================================
# DISEASES
# ============================================================

DISEASES = [
    "Diabetes",
    "Heart Disease",
    "Bronchitis",
    "Breast Cancer"
]


# ============================================================
# FACTOR 1 — LABORATORY REPORTS
# ============================================================

LAB_FEATURES = [
    "Glucose",
    "HbA1c",
    "Cholesterol",
    "HDL",
    "LDL",
    "Triglycerides",
    "Hemoglobin",
    "Creatinine",
    "WBC",
    "Platelets"
]


# ============================================================
# FACTOR 2 — MEDICAL HISTORY
# ============================================================

HISTORY_FEATURES = [
    "Age",
    "Gender",
    "Family_History",
    "Previous_Diseases",
    "Smoking_History",
    "Alcohol_History",
    "Diabetes_History",
    "Heart_Disease_History",
    "Bronchitis_History",
    "Breast_Cancer_History"
]


# ============================================================
# FACTOR 3 — VITAL SIGNS
# ============================================================

VITAL_FEATURES = [
    "Systolic_BP",
    "Diastolic_BP",
    "Heart_Rate",
    "Respiratory_Rate",
    "SpO2",
    "Temperature",
    "BMI",
    "Height_cm",
    "Weight_kg"
]


# ============================================================
# FACTOR 4 — MEDICAL IMAGING
# ============================================================

IMAGING_FEATURES = [
    "Chest_Xray_Finding",
    "Mammogram_Finding",
    "Imaging_Report",
    "Image_Category"
]


# ============================================================
# ALL INPUT FEATURES
# ============================================================

FEATURES = (
    LAB_FEATURES
    + HISTORY_FEATURES
    + VITAL_FEATURES
    + IMAGING_FEATURES
)


# ============================================================
# TARGET COLUMNS
# ============================================================

TARGET_DISEASE = "Disease_Label"

TARGET_RISK = "Risk_Level"

TARGET_NEXT = "Next_Possible_Disease"


# ============================================================
# LOAD DATASET
# ============================================================

def load_data(path=DATA_PATH):
    """
    Load the Excel dataset used by the project.
    """

    if not Path(path).exists():

        raise FileNotFoundError(
            f"Dataset not found at: {path}"
        )

    df = pd.read_excel(
        path,
        sheet_name=0,
        header=1
    )


    # --------------------------------------------------------
    # Remove duplicate columns
    # --------------------------------------------------------

    df = df.loc[
        :,
        ~df.columns.duplicated()
    ].copy()


    # --------------------------------------------------------
    # Remove accidental duplicate next-disease columns
    # --------------------------------------------------------

    duplicate_next_columns = [
        column
        for column in df.columns
        if column.startswith(
            "Next_Possible_Disease_"
        )
    ]


    if duplicate_next_columns:

        df = df.drop(
            columns=duplicate_next_columns
        )


    # --------------------------------------------------------
    # Convert empty text values into NaN
    # --------------------------------------------------------

    empty_values = [
        "",
        " ",
        "nan",
        "NaN",
        "None",
        "none",
        "NULL",
        "null"
    ]


    for column in (
        FEATURES
        + [
            TARGET_DISEASE,
            TARGET_RISK,
            TARGET_NEXT
        ]
    ):

        if column in df.columns:

            if df[column].dtype == object:

                df[column] = df[column].replace(
                    empty_values,
                    np.nan
                )


    return df


# ============================================================
# DATASET VALIDATION
# ============================================================

def validate_dataset(df):
    """
    Check whether all required columns exist.
    """

    required_columns = (
        FEATURES
        + [
            TARGET_DISEASE,
            TARGET_RISK,
            TARGET_NEXT
        ]
    )


    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]


    return missing_columns


# ============================================================
# GET NUMERIC AND CATEGORICAL FEATURES
# ============================================================

def get_feature_types(df):

    numeric_features = []

    categorical_features = []


    for column in FEATURES:

        if column not in df.columns:
            continue


        if pd.api.types.is_numeric_dtype(
            df[column]
        ):

            numeric_features.append(column)

        else:

            categorical_features.append(column)


    return (
        numeric_features,
        categorical_features
    )


# ============================================================
# CREATE PREPROCESSING PIPELINE
# ============================================================

def make_preprocessor(df):

    numeric_features, categorical_features = (
        get_feature_types(df)
    )


    # --------------------------------------------------------
    # Numerical preprocessing
    # --------------------------------------------------------

    numeric_pipeline = Pipeline([

        (
            "imputer",

            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",

            StandardScaler()
        )

    ])


    # --------------------------------------------------------
    # Categorical preprocessing
    # --------------------------------------------------------

    categorical_pipeline = Pipeline([

        (
            "imputer",

            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "onehot",

            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )

    ])


    # --------------------------------------------------------
    # Combine both pipelines
    # --------------------------------------------------------

    preprocessor = ColumnTransformer([

        (
            "num",

            numeric_pipeline,

            numeric_features
        ),

        (
            "cat",

            categorical_pipeline,

            categorical_features
        )

    ], remainder="drop")


    return (
        preprocessor,
        numeric_features,
        categorical_features
    )


# ============================================================
# GET PROCESSED FEATURE NAMES
# ============================================================

def feature_names_from_preprocessor(
    preprocessor
):

    return list(
        preprocessor
        .get_feature_names_out()
    )