from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

from preprocessing import (
    load_data,
    validate_dataset,
    make_preprocessor,
    FEATURES,
    TARGET_DISEASE,
    TARGET_RISK,
    TARGET_NEXT
)

warnings.filterwarnings("ignore")


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


# ============================================================
# RANDOM STATE
# ============================================================

RANDOM_STATE = 42


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def get_models():

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=3000,
            C=1.0,
            solver="lbfgs"
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=RANDOM_STATE
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=RANDOM_STATE
        )
    }

    return models


# ============================================================
# CHECK CLASS BALANCE
# ============================================================

def check_class_balance(y):

    counts = pd.Series(y).value_counts()

    if len(counts) == 0:
        return False

    largest = counts.max()
    smallest = counts.min()

    imbalance_ratio = largest / smallest

    return imbalance_ratio > 1.5


# ============================================================
# TRAIN ONE TARGET
# ============================================================

def train_target(
    df,
    target,
    filename,
    label_encoder_filename
):

    print("\n")
    print("=" * 80)
    print(f"TRAINING TARGET: {target}")
    print("=" * 80)

    # --------------------------------------------------------
    # Remove rows where target is missing
    # --------------------------------------------------------

    data = df.dropna(subset=[target]).copy()

    print(f"Total usable rows: {len(data)}")

    # --------------------------------------------------------
    # Input features
    # --------------------------------------------------------

    X = data[FEATURES].copy()

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    y_raw = data[target].astype(str)

    print("\nClass distribution:")

    class_counts = y_raw.value_counts()

    for cls, count in class_counts.items():
        percentage = (count / len(y_raw)) * 100
        print(
            f"  {cls}: {count} samples "
            f"({percentage:.2f}%)"
        )

    # --------------------------------------------------------
    # Encode target labels
    # --------------------------------------------------------

    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(y_raw)

    print("\nEncoded classes:")

    for index, cls in enumerate(label_encoder.classes_):
        print(f"  {index} -> {cls}")

    # --------------------------------------------------------
    # Train / test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nDataset split:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Testing samples : {len(X_test)}")

    # --------------------------------------------------------
    # Preprocessor
    # --------------------------------------------------------

    preprocessor, numeric_features, categorical_features = (
        make_preprocessor(data)
    )

    print("\nFeature information:")
    print(f"  Numeric features    : {len(numeric_features)}")
    print(f"  Categorical features: {len(categorical_features)}")

    # --------------------------------------------------------
    # Class balance
    # --------------------------------------------------------

    imbalance = check_class_balance(y_train)

    if imbalance:
        print("\nClass imbalance detected.")
        print("Balanced class weights will be used where supported.")
    else:
        print("\nClasses are reasonably balanced.")
        print("Normal class weighting will be used.")

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    candidates = get_models()

    # Apply class weighting only when genuinely necessary.
    if imbalance:

        candidates["Logistic Regression"] = LogisticRegression(
            max_iter=3000,
            C=1.0,
            solver="lbfgs",
            class_weight="balanced"
        )

        candidates["Random Forest"] = RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        candidates["Extra Trees"] = ExtraTreesClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    # --------------------------------------------------------
    # Train and evaluate each model
    # --------------------------------------------------------

    results = []
    fitted_models = {}

    for name, estimator in candidates.items():

        print("\n" + "-" * 70)
        print(f"Training: {name}")
        print("-" * 70)

        try:

            pipe = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", estimator)
                ]
            )

            pipe.fit(X_train, y_train)

            predictions = pipe.predict(X_test)

            accuracy = accuracy_score(
                y_test,
                predictions
            )

            precision, recall, f1, _ = (
                precision_recall_fscore_support(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                )
            )

            result = {
                "model": name,
                "accuracy": float(accuracy),
                "precision_weighted": float(precision),
                "recall_weighted": float(recall),
                "f1_weighted": float(f1)
            }

            results.append(result)

            fitted_models[name] = (
                pipe,
                predictions
            )

            print(
                f"Accuracy : {accuracy * 100:.2f}%"
            )

            print(
                f"Precision: {precision * 100:.2f}%"
            )

            print(
                f"Recall   : {recall * 100:.2f}%"
            )

            print(
                f"F1 Score : {f1 * 100:.2f}%"
            )

        except Exception as e:

            print(
                f"ERROR while training {name}: {e}"
            )

    # --------------------------------------------------------
    # Make sure at least one model succeeded
    # --------------------------------------------------------

    if not results:

        raise RuntimeError(
            f"No model could be successfully trained for target: {target}"
        )

    # --------------------------------------------------------
    # Select best model
    #
    # F1 is used because it balances precision and recall.
    # --------------------------------------------------------

    best_result = max(
        results,
        key=lambda x: x["f1_weighted"]
    )

    best_name = best_result["model"]

    best_pipe = fitted_models[best_name][0]

    best_predictions = fitted_models[best_name][1]

    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    model_path = MODEL_DIR / filename

    encoder_path = (
        MODEL_DIR / label_encoder_filename
    )

    joblib.dump(
        best_pipe,
        model_path
    )

    joblib.dump(
        label_encoder,
        encoder_path
    )

    print("\nBest model:")
    print(f"  {best_name}")

    print(
        f"  Accuracy : "
        f"{best_result['accuracy'] * 100:.2f}%"
    )

    print(
        f"  Precision: "
        f"{best_result['precision_weighted'] * 100:.2f}%"
    )

    print(
        f"  Recall   : "
        f"{best_result['recall_weighted'] * 100:.2f}%"
    )

    print(
        f"  F1 Score : "
        f"{best_result['f1_weighted'] * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        best_predictions
    ).tolist()

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        y_test,
        best_predictions,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0
    )

    # --------------------------------------------------------
    # Return complete metrics
    # --------------------------------------------------------

    return {

        "target": target,

        "best_model": best_name,

        "classes": (
            label_encoder.classes_.tolist()
        ),

        "results": results,

        "best_accuracy": float(
            best_result["accuracy"]
        ),

        "best_precision": float(
            best_result["precision_weighted"]
        ),

        "best_recall": float(
            best_result["recall_weighted"]
        ),

        "best_f1": float(
            best_result["f1_weighted"]
        ),

        "confusion_matrix": cm,

        "classification_report": report,

        "test_size": int(len(y_test)),

        "train_size": int(len(y_train)),

        "class_distribution": {
            str(k): int(v)
            for k, v in class_counts.items()
        }
    }


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def main():

    print("\n")
    print("=" * 80)
    print("MULTIMODAL HEALTH RISK PREDICTION")
    print("MODEL TRAINING")
    print("=" * 80)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading dataset...")

    df = load_data()

    print(
        f"Dataset loaded successfully."
    )

    print(
        f"Rows    : {len(df)}"
    )

    print(
        f"Columns : {len(df.columns)}"
    )

    # --------------------------------------------------------
    # Validate dataset
    # --------------------------------------------------------

    print("\nValidating dataset...")

    missing = validate_dataset(df)

    if missing:

        raise ValueError(
            "The following required columns are missing:\n"
            + "\n".join(
                f" - {column}"
                for column in missing
            )
        )

    print(
        "Dataset validation successful."
    )

    # ========================================================
    # 1. DISEASE MODEL
    # ========================================================

    disease = train_target(
        df,
        TARGET_DISEASE,
        "disease_model.joblib",
        "disease_label_encoder.joblib"
    )

    # ========================================================
    # 2. RISK MODEL
    # ========================================================

    risk = train_target(
        df,
        TARGET_RISK,
        "risk_model.joblib",
        "risk_label_encoder.joblib"
    )

    # ========================================================
    # 3. NEXT DISEASE MODEL
    # ========================================================

    next_disease = train_target(
        df,
        TARGET_NEXT,
        "next_disease_model.joblib",
        "next_disease_label_encoder.joblib"
    )

    # ========================================================
    # CREATE COMPLETE METRICS
    # ========================================================

    metrics = {

        "dataset_rows": int(len(df)),

        "dataset_columns": int(len(df.columns)),

        "features_used": FEATURES,

        "disease_model": disease,

        "risk_model": risk,

        "next_disease_model": next_disease
    }

    # ========================================================
    # SAVE METRICS.JSON
    # ========================================================

    metrics_path = (
        REPORT_DIR / "metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)

    print("\nDISEASE MODEL")
    print("-" * 80)

    print(
        f"Best Model : "
        f"{disease['best_model']}"
    )

    print(
        f"Accuracy   : "
        f"{disease['best_accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision  : "
        f"{disease['best_precision'] * 100:.2f}%"
    )

    print(
        f"Recall     : "
        f"{disease['best_recall'] * 100:.2f}%"
    )

    print(
        f"F1 Score   : "
        f"{disease['best_f1'] * 100:.2f}%"
    )

    print("\nRISK MODEL")
    print("-" * 80)

    print(
        f"Best Model : "
        f"{risk['best_model']}"
    )

    print(
        f"Accuracy   : "
        f"{risk['best_accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision  : "
        f"{risk['best_precision'] * 100:.2f}%"
    )

    print(
        f"Recall     : "
        f"{risk['best_recall'] * 100:.2f}%"
    )

    print(
        f"F1 Score   : "
        f"{risk['best_f1'] * 100:.2f}%"
    )

    print("\nNEXT DISEASE MODEL")
    print("-" * 80)

    print(
        f"Best Model : "
        f"{next_disease['best_model']}"
    )

    print(
        f"Accuracy   : "
        f"{next_disease['best_accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision  : "
        f"{next_disease['best_precision'] * 100:.2f}%"
    )

    print(
        f"Recall     : "
        f"{next_disease['best_recall'] * 100:.2f}%"
    )

    print(
        f"F1 Score   : "
        f"{next_disease['best_f1'] * 100:.2f}%"
    )

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    print("\n")
    print("=" * 80)
    print("DISEASE MODEL COMPARISON")
    print("=" * 80)

    for result in disease["results"]:

        print(
            f"{result['model']:<25}"
            f" Accuracy: "
            f"{result['accuracy'] * 100:>7.2f}%"
            f" | F1: "
            f"{result['f1_weighted'] * 100:>7.2f}%"
        )

    print("\n")
    print("=" * 80)
    print("RISK MODEL COMPARISON")
    print("=" * 80)

    for result in risk["results"]:

        print(
            f"{result['model']:<25}"
            f" Accuracy: "
            f"{result['accuracy'] * 100:>7.2f}%"
            f" | F1: "
            f"{result['f1_weighted'] * 100:>7.2f}%"
        )

    print("\n")
    print("=" * 80)
    print("NEXT DISEASE MODEL COMPARISON")
    print("=" * 80)

    for result in next_disease["results"]:

        print(
            f"{result['model']:<25}"
            f" Accuracy: "
            f"{result['accuracy'] * 100:>7.2f}%"
            f" | F1: "
            f"{result['f1_weighted'] * 100:>7.2f}%"
        )

    # ========================================================
    # FILE LOCATIONS
    # ========================================================

    print("\n")
    print("=" * 80)
    print("FILES GENERATED")
    print("=" * 80)

    print(
        f"\nModels saved in:\n"
        f"{MODEL_DIR}"
    )

    print(
        f"\nMetrics saved in:\n"
        f"{metrics_path}"
    )

    print("\n")
    print("=" * 80)
    print("ALL TRAINING TASKS FINISHED SUCCESSFULLY")
    print("=" * 80)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
    