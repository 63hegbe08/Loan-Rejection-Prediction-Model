

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score

RANDOM_STATE = 42
TARGET = "TURNDOWN"


NUMERIC_COLS = [
    "AGE", "EDUC", "KIDS", "INCOME", "NETWORTH", "ASSET", "DEBT",
    "DEBT2INC", "CCBAL", "HOMEEQ",
    "MARRIED", "LF", "HHSEX", "LATE", "LATE60", "BNKRUPLAST5", "FORECLLAST5",
]

CATEGORICAL_COLS = ["RACE"]

FEATURE_COLUMNS = NUMERIC_COLS + CATEGORICAL_COLS


def build_pipeline(estimator):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUMERIC_COLS),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), CATEGORICAL_COLS),
        ]
    )
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", estimator),
    ])


def train_and_save(csv_path: str = "SCFP2019.csv", model_path: str = "model.pkl"):
    df = pd.read_csv(csv_path)

   
    df = df[df["Y1"] % 10 == 1].copy()

    
    df = df[df["CRDAPP"] == 1].copy()

    X = df[FEATURE_COLUMNS]
    y = df[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    cv_scores = {}
    for name, est in candidates.items():
        pipe = build_pipeline(est)
        scores = cross_val_score(pipe, x_train, y_train, cv=5, scoring="roc_auc")
        cv_scores[name] = scores.mean()
        print(f"{name:18s}: ROC-AUC = {scores.mean():.4f} (+/- {scores.std():.4f})")

    best_name = max(cv_scores, key=cv_scores.get)
    print(f"\nBest candidate by CV: {best_name}")

    param_grids = {
        "LogisticRegression": {
            "model__C": [0.1, 1.0, 10.0],
        },
        "RandomForest": {
            "model__n_estimators": [200, 400],
            "model__max_depth": [4, 6, 8, None],
            "model__min_samples_leaf": [1, 2, 4],
        },
        "GradientBoosting": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [2, 3, 4],
            "model__learning_rate": [0.03, 0.05, 0.1],
        },
    }

    best_pipeline = build_pipeline(candidates[best_name])
    grid = param_grids.get(best_name, {})

    if grid:
        search = GridSearchCV(
            estimator=best_pipeline,
            param_grid=grid,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1,
        )
        search.fit(x_train, y_train)
        best_pipeline = search.best_estimator_
        print(f"Best parameters for {best_name}:", search.best_params_)
    else:
        best_pipeline.fit(x_train, y_train)

    y_pred = best_pipeline.predict(x_test)
    y_proba = best_pipeline.predict_proba(x_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"\nModel        : {best_name}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test ROC-AUC : {roc_auc:.4f}")
    print(f"Precision    : {precision:.4f}")
    print(f"Recall       : {recall:.4f}  <- how many actual rejections we catch")

   
    numeric_stats = {
        col: {"min": float(df[col].min()), "max": float(df[col].max()), "mean": float(df[col].mean())}
        for col in NUMERIC_COLS
    }
    categorical_options = {
        col: sorted(df[col].dropna().unique().tolist()) for col in CATEGORICAL_COLS
    }

    joblib.dump(
        {
            "pipeline": best_pipeline,
            "model_name": best_name,
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "precision": precision,
            "recall": recall,
            "feature_columns": FEATURE_COLUMNS,
            "numeric_stats": numeric_stats,
            "categorical_options": categorical_options,
        },
        model_path,
    )
    print(f"\nModel and metadata saved to '{model_path}'")


if __name__ == "__main__":
    train_and_save()
