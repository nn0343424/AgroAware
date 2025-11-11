# ml_service/tune_hparams.py
import warnings, json
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, f1_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

import optuna

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
MODELS_DIR = APP_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

TRAIN_CSV = DATA_DIR / "Train Dataset.csv"
FEATURES = ["N", "P", "K", "ph", "temperature", "rainfall"]
LABEL_CANDIDATES = ["label", "Label", "crop", "Crop", "target", "Target"]

def load_train():
    if not TRAIN_CSV.exists():
        raise FileNotFoundError("Train Dataset.csv not found.")
    df = pd.read_csv(TRAIN_CSV)
    df.columns = [c.strip() for c in df.columns]
    y_col = None
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            y_col = c
            break
    if y_col is None:
        raise KeyError(f"Could not find label column among {LABEL_CANDIDATES}")
    X = df[FEATURES].astype(float).values
    y_raw = df[y_col].astype(str).values
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    enc = LabelEncoder().fit(y_raw)
    y = enc.transform(y_raw)
    return Xs, y

X, y = load_train()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scorer = make_scorer(f1_score, average="macro")

def tune_decision_tree(trial):
    params = {
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        "max_depth": trial.suggest_int("max_depth", 3, 40),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "random_state": 42,
    }
    clf = DecisionTreeClassifier(**params)
    return np.mean(cross_val_score(clf, X, y, cv=cv, scoring=scorer))

def tune_random_forest(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 5, 40),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "random_state": 42,
        "n_jobs": -1,
    }
    clf = RandomForestClassifier(**params)
    return np.mean(cross_val_score(clf, X, y, cv=cv, scoring=scorer))

def tune_logreg(trial):
    params = {
        "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
        "penalty": trial.suggest_categorical("penalty", ["l2"]),
        "solver": trial.suggest_categorical("solver", ["lbfgs", "saga"]),
        "max_iter": 2000,
        "random_state": 42,
        "n_jobs": None,
    }
    clf = LogisticRegression(**params)
    return np.mean(cross_val_score(clf, X, y, cv=cv, scoring=scorer))

def tune_xgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "random_state": 42,
        "eval_metric": "mlogloss",
        "n_jobs": -1,
    }
    clf = xgb.XGBClassifier(**params)
    return np.mean(cross_val_score(clf, X, y, cv=cv, scoring=scorer))

SEARCH = [
    ("DecisionTree", tune_decision_tree, 60),
    ("RandomForest", tune_random_forest, 80),
    ("LogisticRegression", tune_logreg, 60),
    ("XGBoost", tune_xgb, 80),
]

def main():
    results = {}
    best_overall = (-1.0, None)  # (score, name)

    for name, fn, n_trials in SEARCH:
        print(f"🔎 Tuning {name} ({n_trials} trials)...")
        study = optuna.create_study(direction="maximize")
        study.optimize(fn, n_trials=n_trials, show_progress_bar=True)
        best_val = float(study.best_value)
        best_params = dict(study.best_params)
        print(f"  -> Best {name}: {best_val:.4f}")
        results[name] = {"best_value": best_val, "best_params": best_params}
        if best_val > best_overall[0]:
            best_overall = (best_val, name)

    payload = {
        "best_model": best_overall[1],
        "best_cv_macro_f1": best_overall[0],
        "details": results,
    }
    with open(MODELS_DIR / "tuned_params.json", "w") as f:
        json.dump(payload, f, indent=2)
    print("✅ Saved tuned params to models/tuned_params.json")

if __name__ == "__main__":
    main()
