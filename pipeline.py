"""
pipeline.py - Scikit-Learn Pipeline + MLflow Experiment Tracking
DTSC6012001 Model Deployment - Mid Exam 2026
Dataset A (Fitur & Target terpisah, NIM Ganjil)
"""

import os
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score,
    mean_squared_error, r2_score, mean_absolute_error
)

import mlflow
import mlflow.sklearn

# 1. DATA INGESTION - Modular function

def load_data(features_path: str, targets_path: str) -> pd.DataFrame:
    """
    Load dan merge dataset fitur + target.
    Menghapus kolom Student_ID setelah merge.
    """
    features = pd.read_csv(features_path)
    targets  = pd.read_csv(targets_path)
    df = features.merge(targets, on='Student_ID')
    df.drop(columns=['Student_ID'], inplace=True)
    print(f"[INFO] Dataset loaded: {df.shape[0]} rows x {df.shape[1]} cols")
    return df


def prepare_features(df: pd.DataFrame):
    """
    Pisahkan fitur & target, identifikasi kolom numerik & kategorikal.
    Returns: X, y_clf, y_reg, num_cols, cat_cols
    """
    TARGET_CLF = 'placement_status'
    TARGET_REG = 'salary_lpa'

    y_clf = (df[TARGET_CLF] == 'Placed').astype(int)
    y_reg = df[TARGET_REG]

    X = df.drop(columns=[TARGET_CLF, TARGET_REG])

    num_cols = X.select_dtypes(include='number').columns.tolist()
    cat_cols = X.select_dtypes(include='object').columns.tolist()

    return X, y_clf, y_reg, num_cols, cat_cols


# 2. PIPELINE BUILDER - End-to-End (preprocessing + model)

def build_clf_pipeline(num_cols, cat_cols, model) -> Pipeline:
    """
    Bangun pipeline klasifikasi: impute, scale/encode, model.
    Semua preprocessing terintegrasi untuk mencegah data leakage.
    """
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    return pipeline


def build_reg_pipeline(num_cols, cat_cols, model) -> Pipeline:
    """
    Bangun pipeline regresi: impute, scale/encode, model.
    """
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    return pipeline


# 3. EXPERIMENT TRACKING WITH MLFLOW

def run_classification_experiments(X_train, X_test, y_train, y_test, num_cols, cat_cols):
    """
    Jalankan eksperimen klasifikasi dengan MLflow tracking.
    Log parameter, metrik, dan simpan artifact model.
    """
    mlflow.set_experiment("placement_classification")

    clf_configs = [
        {
            'name': 'LogisticRegression',
            'model': LogisticRegression(max_iter=500, C=1.0, random_state=42),
            'params': {'C': 1.0, 'max_iter': 500}
        },
        {
            'name': 'RandomForestClassifier',
            'model': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            'params': {'n_estimators': 100, 'max_depth': 10}
        },
        {
            'name': 'GradientBoostingClassifier',
            'model': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
            'params': {'n_estimators': 100, 'learning_rate': 0.1}
        }
    ]

    best_pipeline = None
    best_accuracy = 0
    best_name = ''

    for cfg in clf_configs:
        with mlflow.start_run(run_name=cfg['name']):
            pipeline = build_clf_pipeline(num_cols, cat_cols, cfg['model'])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1  = f1_score(y_test, y_pred, average='weighted')

            # Log ke MLflow
            mlflow.log_params(cfg['params'])
            mlflow.log_metric('accuracy', acc)
            mlflow.log_metric('f1_score', f1)
            mlflow.sklearn.log_model(pipeline, artifact_path='model')

            print(f"[CLF] {cfg['name']}: accuracy={acc:.4f}, f1={f1:.4f}")

            if acc > best_accuracy:
                best_accuracy = acc
                best_pipeline = pipeline
                best_name = cfg['name']

    print(f"\n Best Classification Model: {best_name} (accuracy={best_accuracy:.4f})")
    return best_pipeline, best_name


def run_regression_experiments(X_train, X_test, y_train, y_test, num_cols, cat_cols):
    """
    Jalankan eksperimen regresi dengan MLflow tracking.
    """
    mlflow.set_experiment("salary_regression")

    reg_configs = [
        {
            'name': 'LinearRegression',
            'model': LinearRegression(),
            'params': {'fit_intercept': True}
        },
        {
            'name': 'RandomForestRegressor',
            'model': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            'params': {'n_estimators': 100, 'max_depth': 10}
        },
        {
            'name': 'GradientBoostingRegressor',
            'model': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
            'params': {'n_estimators': 100, 'learning_rate': 0.1}
        }
    ]

    best_pipeline = None
    best_r2 = -np.inf
    best_name = ''

    for cfg in reg_configs:
        with mlflow.start_run(run_name=cfg['name']):
            pipeline = build_reg_pipeline(num_cols, cat_cols, cfg['model'])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae  = mean_absolute_error(y_test, y_pred)
            r2   = r2_score(y_test, y_pred)

            # Log ke MLflow
            mlflow.log_params(cfg['params'])
            mlflow.log_metric('rmse', rmse)
            mlflow.log_metric('mae', mae)
            mlflow.log_metric('r2', r2)
            mlflow.sklearn.log_model(pipeline, artifact_path='model')

            print(f"[REG] {cfg['name']}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

            if r2 > best_r2:
                best_r2 = r2
                best_pipeline = pipeline
                best_name = cfg['name']

    print(f"\n Best Regression Model: {best_name} (R²={best_r2:.4f})")
    return best_pipeline, best_name


# 4. PERSISTENCE - Simpan model terbaik

def save_models(clf_pipeline, reg_pipeline, clf_name, reg_name,
                output_path='best_models.pkl'):
    """
    Simpan kedua pipeline terbaik ke file .pkl untuk digunakan deployment.
    """
    payload = {
        'clf_pipeline': clf_pipeline,
        'reg_pipeline': reg_pipeline,
        'clf_name':     clf_name,
        'reg_name':     reg_name
    }
    with open(output_path, 'wb') as f:
        pickle.dump(payload, f)
    print(f"\n Models saved to '{output_path}'")


# MAIN

if __name__ == '__main__':
    # Path ke dataset — sesuaikan jika perlu
    FEATURES_PATH = 'A.csv'
    TARGETS_PATH  = 'A_targets.csv'

    # 1. Load data
    df = load_data(FEATURES_PATH, TARGETS_PATH)

    # 2. Prepare features
    X, y_clf, y_reg, num_cols, cat_cols = prepare_features(df)

    # 3. Train-test split 80:20
    X_train, X_test, \
    y_clf_train, y_clf_test, \
    y_reg_train, y_reg_test = train_test_split(
        X, y_clf, y_reg, test_size=0.2, random_state=42
    )
    print(f"[INFO] Train: {X_train.shape}, Test: {X_test.shape}")

    # 4. Run experiments dengan MLflow
    print("\n=== Classification Experiments ===")
    best_clf, clf_name = run_classification_experiments(
        X_train, X_test, y_clf_train, y_clf_test, num_cols, cat_cols
    )

    print("\n=== Regression Experiments ===")
    best_reg, reg_name = run_regression_experiments(
        X_train, X_test, y_reg_train, y_reg_test, num_cols, cat_cols
    )

    # 5. Simpan model
    save_models(best_clf, best_reg, clf_name, reg_name)

    print("\n Pipeline selesai!")
