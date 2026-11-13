from pathlib import Path

import numpy as np
import pandas as pd
import shap


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# LOAD PROJECT PREPROCESSING
# ============================================================

try:
    from .preprocessing import load_data, FEATURES
except ImportError:
    from preprocessing import load_data, FEATURES


# ============================================================
# SHAP BACKGROUND SETTINGS
# ============================================================

# Number of real dataset rows used as the SHAP background.
# This is much better than using the patient's own row.
SHAP_BACKGROUND_SIZE = 150

_shap_background_cache = None


# ============================================================
# GET TRANSFORMED DATA
# ============================================================

def get_transformed_data(pipe, X):
    """
    Transform raw input using the exact preprocessing pipeline
    stored inside the trained model.
    """

    preprocessor = pipe.named_steps["preprocessor"]
    model = pipe.named_steps["model"]

    Xt = preprocessor.transform(X)

    feature_names = list(
        preprocessor.get_feature_names_out()
    )

    return Xt, feature_names, model


# ============================================================
# LOAD REAL DATASET FOR SHAP BACKGROUND
# ============================================================

def get_shap_background():
    """
    Load real project dataset and prepare a representative
    background dataset for SHAP.

    IMPORTANT:
    We DO NOT use the patient's own row as the background.

    A representative set of real records provides SHAP with
    a meaningful reference distribution.
    """

    global _shap_background_cache

    if _shap_background_cache is not None:
        return _shap_background_cache

    try:

        df = load_data()

        available_features = [
            feature
            for feature in FEATURES
            if feature in df.columns
        ]

        if not available_features:
            raise ValueError(
                "No project features were found in the dataset."
            )

        background = (
            df[available_features]
            .copy()
        )

        # Remove completely empty rows.
        background = background.dropna(
            how="all"
        )

        # Use a fixed random sample if dataset is large.
        if len(background) > SHAP_BACKGROUND_SIZE:

            background = background.sample(
                n=SHAP_BACKGROUND_SIZE,
                random_state=42
            )

        background = background.reset_index(
            drop=True
        )

        _shap_background_cache = background

        print(
            f"SHAP background loaded: "
            f"{len(background)} real records"
        )

        return background

    except Exception as error:

        print(
            "Could not load SHAP background dataset:",
            error
        )

        return None


# ============================================================
# EXTRACT SHAP VALUES FOR PREDICTED CLASS
# ============================================================

def _extract_class_values(
    shap_values,
    predicted_index,
    n_features,
    n_classes=None
):
    """
    Normalize SHAP output across different SHAP versions
    and model types.

    Handles:
        list[class] -> (samples, features)
        (samples, features, classes)
        (samples, classes, features)
        (samples, features)
    """

    predicted_index = int(
        max(
            0,
            predicted_index
        )
    )


    # ========================================================
    # OLD SHAP FORMAT
    # ========================================================

    if isinstance(
        shap_values,
        list
    ):

        if not shap_values:

            return np.zeros(
                n_features,
                dtype=float
            )

        class_index = min(
            predicted_index,
            len(shap_values) - 1
        )

        arr = np.asarray(
            shap_values[class_index]
        )

        if arr.ndim == 2:

            return np.asarray(
                arr[0],
                dtype=float
            ).ravel()

        if arr.ndim == 1:

            return np.asarray(
                arr,
                dtype=float
            ).ravel()


    # ========================================================
    # NEW SHAP FORMAT
    # ========================================================

    arr = np.asarray(
        shap_values
    )


    # --------------------------------------------------------
    # Three-dimensional output
    # --------------------------------------------------------

    if arr.ndim == 3:

        # (samples, features, classes)

        if (
            arr.shape[1] == n_features
            and arr.shape[2] > 1
        ):

            class_index = min(
                predicted_index,
                arr.shape[2] - 1
            )

            return np.asarray(
                arr[
                    0,
                    :,
                    class_index
                ],
                dtype=float
            ).ravel()


        # (samples, classes, features)

        if (
            arr.shape[2] == n_features
            and arr.shape[1] > 1
        ):

            class_index = min(
                predicted_index,
                arr.shape[1] - 1
            )

            return np.asarray(
                arr[
                    0,
                    class_index,
                    :
                ],
                dtype=float
            ).ravel()


        return np.asarray(
            arr[0],
            dtype=float
        ).reshape(-1)[:n_features]


    # --------------------------------------------------------
    # Two-dimensional output
    # --------------------------------------------------------

    if arr.ndim == 2:

        values = np.asarray(
            arr[0],
            dtype=float
        ).ravel()

        # Binary classification:
        #
        # Some SHAP versions return values for the positive
        # class only.
        #
        # If class 0 is being explained, reverse the direction.
        if (
            n_classes == 2
            and predicted_index == 0
        ):

            values = -values

        return values


    # --------------------------------------------------------
    # One-dimensional output
    # --------------------------------------------------------

    if arr.ndim == 1:

        values = np.asarray(
            arr,
            dtype=float
        ).ravel()

        if (
            n_classes == 2
            and predicted_index == 0
        ):

            values = -values

        return values


    return np.asarray(
        arr,
        dtype=float
    ).reshape(-1)


# ============================================================
# RAW FEATURE MAPPING
# ============================================================

def _raw_feature_mapping(
    pipe,
    transformed_names
):
    """
    Map transformed pipeline features back to the original
    user-facing feature names.

    Examples:

        num__Glucose
            -> Glucose

        cat__Gender_Male
            -> Gender

        cat__Smoking_History_Yes
            -> Smoking_History
    """

    preprocessor = (
        pipe.named_steps[
            "preprocessor"
        ]
    )

    try:

        raw_features = [
            str(feature)
            for feature
            in preprocessor.feature_names_in_
        ]

    except Exception:

        raw_features = []


    mapping = []


    for transformed_name in transformed_names:

        clean_name = str(
            transformed_name
        )


        # Remove transformer prefix.

        if "__" in clean_name:

            clean_name = clean_name.split(
                "__",
                1
            )[1]


        matched = None


        # ----------------------------------------------------
        # Exact numerical feature
        # ----------------------------------------------------

        for raw_feature in raw_features:

            if clean_name == raw_feature:

                matched = raw_feature

                break


        # ----------------------------------------------------
        # One-hot categorical feature
        # ----------------------------------------------------

        if matched is None:

            candidates = [

                raw_feature

                for raw_feature
                in raw_features

                if clean_name.startswith(
                    raw_feature + "_"
                )

            ]


            if candidates:

                matched = max(
                    candidates,
                    key=len
                )


        mapping.append(

            matched

            if matched is not None

            else clean_name

        )


    return mapping


# ============================================================
# AGGREGATE SHAP VALUES
# ============================================================

def _aggregate_shap_values(
    values,
    transformed_names,
    pipe
):
    """
    Convert one-hot encoded SHAP values back to the original
    feature names.

    Example:

        Gender_Male
        Gender_Female

    become:

        Gender
    """

    source_names = _raw_feature_mapping(
        pipe,
        transformed_names
    )


    aggregated = {}


    for name, value in zip(
        source_names,
        values
    ):

        try:

            value = float(
                value
            )

        except Exception:

            continue


        if not np.isfinite(value):

            continue


        aggregated[name] = (
            aggregated.get(
                name,
                0.0
            )
            + value
        )


    return list(
        aggregated.items()
    )


# ============================================================
# TREE MODEL DETECTION
# ============================================================

def _is_tree_model(model):

    tree_model_names = {

        "RandomForestClassifier",

        "ExtraTreesClassifier",

        "GradientBoostingClassifier",

        "HistGradientBoostingClassifier"

    }


    return (
        model.__class__.__name__
        in tree_model_names
    )


# ============================================================
# CALCULATE SHAP VALUES
# ============================================================

def _calculate_shap_values(
    pipe,
    Xt,
    model,
    feature_names,
    predicted_index
):
    """
    Calculate SHAP values using the correct explainer.

    Tree models:
        TreeExplainer

    Logistic Regression:
        LinearExplainer + REAL DATASET BACKGROUND

    Other models:
        Generic SHAP explainer + REAL BACKGROUND
    """

    # Number of output classes.

    try:

        n_classes = len(
            model.classes_
        )

    except Exception:

        n_classes = None


    # ========================================================
    # TREE MODELS
    # ========================================================

    if _is_tree_model(model):

        print(
            "SHAP: using TreeExplainer for",
            model.__class__.__name__
        )


        explainer = shap.TreeExplainer(
            model
        )


        shap_values = (
            explainer.shap_values(
                Xt
            )
        )


        values = _extract_class_values(
            shap_values,
            predicted_index,
            len(feature_names),
            n_classes
        )


    # ========================================================
    # LOGISTIC REGRESSION
    # ========================================================

    elif (
        model.__class__.__name__
        == "LogisticRegression"
    ):

        print(
            "SHAP: using LinearExplainer "
            "with real dataset background"
        )


        background_raw = (
            get_shap_background()
        )


        if background_raw is not None:

            background_Xt = (
                pipe.named_steps[
                    "preprocessor"
                ].transform(
                    background_raw
                )
            )

            # Limit background to avoid unnecessary memory
            # usage.

            if len(background_Xt) > SHAP_BACKGROUND_SIZE:

                background_Xt = (
                    background_Xt[
                        :SHAP_BACKGROUND_SIZE
                    ]
                )

        else:

            # This is only a fallback.
            #
            # It is intentionally NOT the patient row alone.
            background_Xt = Xt


        explainer = shap.LinearExplainer(
            model,
            background_Xt
        )


        shap_values = (
            explainer.shap_values(
                Xt
            )
        )


        values = _extract_class_values(
            shap_values,
            predicted_index,
            len(feature_names),
            n_classes
        )


    # ========================================================
    # OTHER MODELS
    # ========================================================

    else:

        print(
            "SHAP: using generic Explainer"
        )


        background_raw = (
            get_shap_background()
        )


        if background_raw is not None:

            background_Xt = (
                pipe.named_steps[
                    "preprocessor"
                ].transform(
                    background_raw
                )
            )

        else:

            background_Xt = Xt


        masker = shap.maskers.Independent(
            background_Xt
        )


        explainer = shap.Explainer(
            model,
            masker,
            feature_names=feature_names
        )


        explanation = explainer(
            Xt
        )


        values = _extract_class_values(
            explanation.values,
            predicted_index,
            len(feature_names),
            n_classes
        )


    # ========================================================
    # DIMENSION SAFETY
    # ========================================================

    values = np.asarray(
        values,
        dtype=float
    ).ravel()


    if len(values) < len(
        feature_names
    ):

        values = np.pad(
            values,
            (
                0,
                len(feature_names)
                - len(values)
            ),
            constant_values=0.0
        )


    if len(values) > len(
        feature_names
    ):

        values = values[
            :len(feature_names)
        ]


    # Replace invalid numerical values.

    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


    return values


# ============================================================
# MAIN SHAP EXPLANATION
# ============================================================

def shap_explanation(
    pipe,
    X_one,
    predicted_index,
    top_n=10
):
    """
    Return the ten most important ORIGINAL input features
    for the predicted disease.

    Output:

        {
            "feature": "Glucose",
            "impact": 0.123456,
            "direction": "increases"
        }
    """

    try:

        # ----------------------------------------------------
        # Transform patient's input
        # ----------------------------------------------------

        (
            Xt,
            transformed_names,
            model
        ) = get_transformed_data(
            pipe,
            X_one
        )


        # ----------------------------------------------------
        # Calculate SHAP
        # ----------------------------------------------------

        values = _calculate_shap_values(
            pipe,
            Xt,
            model,
            transformed_names,
            predicted_index
        )


        # ----------------------------------------------------
        # Aggregate to original features
        # ----------------------------------------------------

        aggregated_pairs = (
            _aggregate_shap_values(
                values,
                transformed_names,
                pipe
            )
        )


        # ----------------------------------------------------
        # Sort by absolute contribution
        # ----------------------------------------------------

        aggregated_pairs.sort(

            key=lambda item:
                abs(
                    float(
                        item[1]
                    )
                ),

            reverse=True

        )


        results = []


        # ----------------------------------------------------
        # Return top 10
        # ----------------------------------------------------

        selected_pairs = (
            aggregated_pairs
            if top_n is None
            else aggregated_pairs[:top_n]
        )

        for feature, impact in selected_pairs:
            impact = float(
                impact
            )


            if not np.isfinite(
                impact
            ):

                continue


            # Do NOT round to zero here.
            #
            # Keep six decimal places so that small but real
            # contributions are visible.

            if abs(impact) < 1e-12:

                direction = (
                    "minimal impact"
                )

            elif impact > 0:

                direction = (
                    "increases"
                )

            else:

                direction = (
                    "decreases"
                )


            results.append({

                "feature":
                    str(
                        feature
                    ).replace(
                        "_",
                        " "
                    ),

                "impact":
                    round(
                        impact,
                        6
                    ),

                "direction":
                    direction

            })


        # ----------------------------------------------------
        # If nothing was generated
        # ----------------------------------------------------

        if not results:

            return [

                {

                    "feature":
                        "No significant factors found",

                    "impact":
                        0.0,

                    "direction":
                        "minimal impact"

                }

            ]


        return results


    except Exception as error:

        print(
            "SHAP explanation unavailable:",
            error
        )


        return [

            {

                "feature":
                    "SHAP unavailable",

                "impact":
                    0.0,

                "direction":
                    "unavailable",

                "error":
                    str(error)

            }

        ]


# ============================================================
# LIME
# ============================================================

def lime_explanation(
    pipe,
    X_train,
    X_one,
    class_names,
    predicted_index
):
    """
    Optional LIME explanation.
    """

    try:

        from lime.lime_tabular import (
            LimeTabularExplainer
        )


        Xt_train, names, _ = (
            get_transformed_data(
                pipe,
                X_train
            )
        )


        Xt_one, _, _ = (
            get_transformed_data(
                pipe,
                X_one
            )
        )


        explainer = LimeTabularExplainer(

            Xt_train,

            feature_names=names,

            class_names=class_names,

            mode="classification",

            discretize_continuous=True,

            random_state=42

        )


        exp = explainer.explain_instance(

            Xt_one[0],

            pipe.named_steps[
                "model"
            ].predict_proba,

            num_features=10

        )


        return [

            {

                "feature":
                    str(feature),

                "weight":
                    round(
                        float(weight),
                        6
                    )

            }

            for feature, weight
            in exp.as_list()

        ]


    except Exception as error:

        return [

            {

                "feature":
                    "LIME unavailable",

                "weight":
                    0,

                "error":
                    str(error)

            }

        ]