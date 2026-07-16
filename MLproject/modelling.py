# ============================================
# modelling.py - Untuk CI / MLflow Project
# ============================================

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import argparse
import os
import warnings
warnings.filterwarnings('ignore')

# --- Argument parser untuk parameter dari MLproject ---
parser = argparse.ArgumentParser()
parser.add_argument('--n_estimators', type=int, default=100)
parser.add_argument('--max_depth', type=int, default=10)
args = parser.parse_args()

# --- Konfigurasi DagsHub ---
# Gunakan environment variable yang diset di GitHub Secrets
DAGSHUB_TOKEN = os.environ.get('DAGSHUB_USER_TOKEN')
if DAGSHUB_TOKEN:
    os.environ['DAGSHUB_USER_TOKEN'] = DAGSHUB_TOKEN
    dagshub.init(repo_owner="SRPakpahanSST", repo_name="titanic-mlflow", mlflow=True)
else:
    # Fallback ke local jika token tidak tersedia (untuk testing lokal)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

print(f"Tracking URI: {mlflow.get_tracking_uri()}")

# --- Load data ---
df = pd.read_csv('namadataset_preprocessing/data_processed.csv')
X = df.drop('survived', axis=1)
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Training dengan MLflow ---
with mlflow.start_run(run_name="CI_RandomForest"):
    # Log parameter dari CLI
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("source", "GitHub_Actions_CI")

    # Training
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Prediksi
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)

    # Log model
    mlflow.sklearn.log_model(model, "random_forest_model")

    # --- Artefak tambahan ---
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Tidak Selamat', 'Selamat'],
                yticklabels=['Tidak Selamat', 'Selamat'])
    plt.title('Confusion Matrix - CI')
    plt.tight_layout()
    plt.savefig('confusion_matrix_ci.png')
    mlflow.log_artifact('confusion_matrix_ci.png')
    plt.close()

    # Feature Importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title('Feature Importance - CI')
    plt.tight_layout()
    plt.savefig('feature_importance_ci.png')
    mlflow.log_artifact('feature_importance_ci.png')
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve - CI')
    ax.legend()
    plt.tight_layout()
    plt.savefig('roc_curve_ci.png')
    mlflow.log_artifact('roc_curve_ci.png')
    plt.close()

    # Simpan model lokal
    joblib.dump(model, 'random_forest_model_ci.pkl')
    mlflow.log_artifact('random_forest_model_ci.pkl')

    print("="*50)
    print("HASIL EVALUASI MODEL (CI)")
    print("="*50)
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print("="*50)
    print("Run selesai. Cek DagsHub untuk hasil eksperimen.")