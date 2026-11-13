
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

from .preprocessing import (
    FEATURES,
    LAB_FEATURES,
    HISTORY_FEATURES,
    VITAL_FEATURES,
    IMAGING_FEATURES
)

from .explainability import shap_explanation


# ============================================================
# PROJECT DIRECTORIES
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"


# ============================================================
# PREVENTION INFORMATION
# ============================================================

PREVENTION = {

    "Diabetes":
        "Maintain a balanced diet, exercise regularly, maintain "
        "a healthy weight, monitor blood glucose, and attend "
        "regular health check-ups.",

    "Heart Disease":
        "Maintain a heart-healthy diet, exercise regularly, "
        "avoid smoking, control blood pressure and cholesterol, "
        "and attend regular check-ups.",

    "Bronchitis":
        "Avoid smoking and polluted environments, stay hydrated, "
        "maintain good respiratory hygiene, and seek medical care "
        "for persistent respiratory symptoms.",

    "Breast Cancer":
        "Follow recommended screening schedules, maintain a healthy "
        "lifestyle, exercise regularly, and discuss unusual breast "
        "changes with a healthcare professional."
}


# ============================================================
# MODALITY DEFINITIONS
# ============================================================

MODALITY_FEATURES = {

    "Laboratory Reports": LAB_FEATURES,

    "Medical History": HISTORY_FEATURES,

    "Vital Signs": VITAL_FEATURES,

    "Medical Imaging": IMAGING_FEATURES

}


# ============================================================
# FACTOR DISPLAY NAMES
# ============================================================

FACTOR_DISPLAY_NAMES = {

    "Laboratory Reports": "Laboratory Reports",

    "Medical History": "Medical History",

    "Vital Signs": "Vital Signs",

    "Medical Imaging": "Imaging-related Information"

}


# ============================================================
# IDENTIFY FEATURE CATEGORY
# ============================================================

def identify_feature_factor(feature_name):

    name = str(feature_name).strip().lower()

    name = name.replace("num__", "")
    name = name.replace("cat__", "")

    name = name.replace("_", " ")

    for factor, features in MODALITY_FEATURES.items():

        for feature in features:

            base = str(feature).strip().lower()

            base = base.replace("_", " ")

            if (
                name == base
                or name.startswith(base + " ")
                or name.startswith(base + "=")
            ):

                return factor

    return None


# ============================================================
# FACTOR-WISE CONTRIBUTION ANALYSIS
# ============================================================

def calculate_factor_contributions(
    shap_items,
    used_modalities
):

    # Only include categories entered by the user.
    factor_totals = {

        factor: 0.0

        for factor in used_modalities

    }

    # Group SHAP values by healthcare category.
    for item in shap_items:

        factor = item.get("factor")

        if factor not in factor_totals:
            continue

        try:

            factor_totals[factor] += abs(
                float(item.get("impact", 0))
            )

        except (ValueError, TypeError):

            continue

    total_impact = sum(factor_totals.values())

    contributions = []

    for factor, magnitude in factor_totals.items():

        percentage = (

            (magnitude / total_impact) * 100

            if total_impact > 0

            else 0.0

        )

        contributions.append({

            "name": FACTOR_DISPLAY_NAMES[factor],

            "percentage": round(percentage, 2),

            "magnitude": round(magnitude, 6)

        })

    # Highest contributing category appears first.
    contributions.sort(

        key=lambda item: item["percentage"],

        reverse=True

    )

    return contributions


# ============================================================
# CHECK EMPTY VALUE
# ============================================================

def is_empty_value(value):

    if value is None:

        return True

    if isinstance(value, str):

        value = value.strip().lower()

        return value in [

            "",

            "select",

            "none",

            "null",

            "nan"

        ]

    try:

        return bool(pd.isna(value))

    except Exception:

        return False


# ============================================================
# DETECT USER PROVIDED MODALITIES
# ============================================================

def detect_used_modalities(payload):

    used_modalities = []

    for modality, features in MODALITY_FEATURES.items():

        for feature in features:

            value = payload.get(feature, "")

            if not is_empty_value(value):

                used_modalities.append(modality)

                break

    return used_modalities


# ============================================================
# VALIDATE INPUT
# ============================================================

def validate_input(payload):

    used_modalities = detect_used_modalities(payload)

    if not used_modalities:

        raise ValueError(

            "Please enter information in at least one "
            "health factor before analyzing."

        )

    return used_modalities


# ============================================================
# CLEAN USER INPUT
# ============================================================

def clean_input(payload):

    row = {}

    for feature in FEATURES:

        value = payload.get(feature, "")

        if is_empty_value(value):

            row[feature] = np.nan

        else:

            row[feature] = value

    # --------------------------------------------------------
    # Numeric features
    # --------------------------------------------------------

    numeric_features = [

        "Age",

        "Glucose",

        "HbA1c",

        "Cholesterol",

        "HDL",

        "LDL",

        "Triglycerides",

        "Hemoglobin",

        "Creatinine",

        "WBC",

        "Platelets",

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

    for feature in numeric_features:

        try:

            if pd.notna(row.get(feature)):

                row[feature] = float(row[feature])

            else:

                row[feature] = np.nan

        except (ValueError, TypeError):

            row[feature] = np.nan

    return pd.DataFrame(

        [row],

        columns=FEATURES

    )


# ============================================================
# LOAD MODELS
# ============================================================

def load_artifacts():

    required_files = [

        "disease_model.joblib",

        "disease_label_encoder.joblib",

        "risk_model.joblib",

        "risk_label_encoder.joblib",

        "next_disease_model.joblib",

        "next_disease_label_encoder.joblib"

    ]

    missing = []

    for filename in required_files:

        if not (MODEL_DIR / filename).exists():

            missing.append(filename)

    if missing:

        raise FileNotFoundError(

            "Missing trained model files:\n"

            + "\n".join(missing)

            + "\n\nRun:\n"

            "python src/train_models.py"

        )

    return {

        "disease":

            joblib.load(

                MODEL_DIR / "disease_model.joblib"

            ),

        "disease_le":

            joblib.load(

                MODEL_DIR / "disease_label_encoder.joblib"

            ),

        "risk":

            joblib.load(

                MODEL_DIR / "risk_model.joblib"

            ),

        "risk_le":

            joblib.load(

                MODEL_DIR / "risk_label_encoder.joblib"

            ),

        "next":

            joblib.load(

                MODEL_DIR / "next_disease_model.joblib"

            ),

        "next_le":

            joblib.load(

                MODEL_DIR / "next_disease_label_encoder.joblib"

            )

    }


# ============================================================
# HUMAN FEATURE NAME
# ============================================================

def human_feature_name(name):

    name = str(name)

    name = name.replace("num__", "")

    name = name.replace("cat__", "")

    name = name.replace("_", " ")

    return name.strip()


# ============================================================
# SHAP RESULTS
# ============================================================

def prepare_shap_results(

    disease_model,

    X,

    disease_idx,

    top_n=10

):

    try:

        shap_items = shap_explanation(

            disease_model,

            X,

            disease_idx,

            top_n=top_n

        )

    except Exception as error:

        print(

            "SHAP explanation unavailable:",

            error

        )

        return []

    results = []

    for item in shap_items:

        try:

            raw_impact = float(

                item.get("impact", 0)

            )

            impact = round(raw_impact, 6)

            raw_feature_name = item.get(

                "feature",

                "Unknown"

            )

            results.append({

                "feature": human_feature_name(
                    raw_feature_name
                ),

                "impact": impact,

                "direction": item.get(
                    "direction",
                    "unknown"
                ),

                "factor": identify_feature_factor(
                    raw_feature_name
                )

            })

        except (ValueError, TypeError):

            continue

    return results


# ============================================================
# MAIN PREDICTION
# ============================================================

def predict(payload):

    # ========================================================
    # VALIDATE INPUT
    # ========================================================

    used_modalities = validate_input(payload)


    # ========================================================
    # LOAD MODELS
    # ========================================================

    artifacts = load_artifacts()


    # ========================================================
    # PREPARE DATA
    # ========================================================

    X = clean_input(payload)


    # ========================================================
    # DISEASE PREDICTION
    # ========================================================

    disease_model = artifacts["disease"]

    disease_prediction = disease_model.predict(X)

    disease_idx = int(disease_prediction[0])

    disease = str(

        artifacts["disease_le"].inverse_transform(

            [disease_idx]

        )[0]

    )


    # ========================================================
    # DISEASE PROBABILITIES
    # ========================================================

    disease_probabilities = {}

    confidence = 0.0

    if hasattr(disease_model, "predict_proba"):

        probabilities = disease_model.predict_proba(X)[0]

        classes = artifacts["disease_le"].classes_

        disease_probabilities = {

            str(cls): round(float(prob), 6)

            for cls, prob in zip(classes, probabilities)

        }

        confidence = float(np.max(probabilities))


    # ========================================================
    # RISK PREDICTION
    # ========================================================

    risk_model = artifacts["risk"]

    risk_prediction = risk_model.predict(X)

    risk_idx = int(risk_prediction[0])

    risk = str(

        artifacts["risk_le"].inverse_transform(

            [risk_idx]

        )[0]

    )


    # ========================================================
    # NEXT POSSIBLE DISEASE
    # ========================================================

    next_model = artifacts["next"]

    next_prediction = next_model.predict(X)

    next_idx = int(next_prediction[0])

    next_disease = str(

        artifacts["next_le"].inverse_transform(

            [next_idx]

        )[0]

    )


    # ========================================================
    # COMPLETE SHAP FEATURE CONTRIBUTIONS
    # ========================================================

    all_shap_items = prepare_shap_results(

        disease_model,

        X,

        disease_idx,

        top_n=None

    )


    # ========================================================
    # FACTOR-WISE CONTRIBUTION
    # ONLY USER-ENTERED CATEGORIES
    # ========================================================

    factor_contributions = calculate_factor_contributions(

        all_shap_items,

        used_modalities

    )


    # ========================================================
    # TOP 10 FEATURE-LEVEL CONTRIBUTIONS
    # ========================================================

    shap_items = all_shap_items[:10]


    # ========================================================
    # PREVENTION
    # ========================================================

    current_prevention = PREVENTION.get(

        disease,

        "Follow routine preventive healthcare "
        "and consult a qualified healthcare professional."

    )

    next_prevention = PREVENTION.get(

        next_disease,

        "Follow routine preventive healthcare "
        "and consult a qualified healthcare professional."

    )


    # ========================================================
    # FINAL RESULT
    # ========================================================

    return {

        "disease": disease,

        "confidence": round(confidence * 100, 2),

        "disease_probabilities": disease_probabilities,

        "risk": risk,

        "next_disease": next_disease,

        "current_prevention": current_prevention,

        "next_prevention": next_prevention,

        "shap": shap_items,

        "factor_contributions": factor_contributions,

        "used_modalities": used_modalities,

        "modality_count": len(used_modalities)

    }
