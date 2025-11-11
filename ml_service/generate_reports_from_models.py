# Script to generate feature importance and confidence histogram from saved models
import joblib
from pathlib import Path
import pandas as pd
import numpy as np

from train import FEATURES

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
MODELS_DIR = APP_DIR / "models"
REPORTS_DIR = APP_DIR / "reports"

TRAIN_CSV = DATA_DIR / "Train Dataset.csv"
TEST_CSV  = DATA_DIR / "Test Dataset.csv"

# Load test data
test_df = pd.read_csv(TEST_CSV)
label_col = None
for c in ["label","Label","crop","Crop","target","Target"]:
    if c in test_df.columns:
        label_col = c
        break
if label_col is None:
    raise SystemExit("Label column not found in test CSV")

X_test = test_df[FEATURES].astype(float).values
y_test_raw = test_df[label_col].astype(str).values

# Load scaler and encoder
scaler = joblib.load(MODELS_DIR / "scaler.pkl")
encoder = joblib.load(MODELS_DIR / "label_encoder.pkl")
X_test_s = scaler.transform(X_test)
y_test = encoder.transform(y_test_raw)

# Load ensemble model
ensemble = joblib.load(MODELS_DIR / "ensemble_model.pkl")

# Build results dict from ensemble.named_estimators_
results = {}
# VotingClassifier may have named_estimators_ or estimators_
named = getattr(ensemble, 'named_estimators_', None)
if named:
    # keys likely are lowercase names used when building
    for key, model in named.items():
        # map to pretty name
        pretty = None
        if 'decision' in key:
            pretty = 'DecisionTree'
        elif 'random' in key:
            pretty = 'RandomForest'
        elif 'logistic' in key or 'logreg' in key:
            pretty = 'LogisticRegression'
        elif 'xgb' in key or 'xgboost' in key:
            pretty = 'XGBoost'
        else:
            pretty = key
        try:
            preds = model.predict(X_test_s)
            acc = (preds == y_test).mean()
        except Exception:
            preds = None
            acc = 0.0
        # try compute macro f1
        from sklearn.metrics import f1_score, precision_score, recall_score
        try:
            if preds is not None:
                f1 = f1_score(y_test, preds, average='macro', zero_division=0)
                prec = precision_score(y_test, preds, average='macro', zero_division=0)
                rec = recall_score(y_test, preds, average='macro', zero_division=0)
            else:
                f1 = prec = rec = 0.0
        except Exception:
            f1 = prec = rec = 0.0
        results[pretty] = {
            'model': model,
            'acc': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'preds': preds
        }
else:
    raise SystemExit('Could not find named_estimators_ in ensemble model')

# Add ensemble entry
try:
    ens_preds = ensemble.predict(X_test_s)
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    ens_acc = accuracy_score(y_test, ens_preds)
    ens_prec = precision_score(y_test, ens_preds, average='macro', zero_division=0)
    ens_rec = recall_score(y_test, ens_preds, average='macro', zero_division=0)
    ens_f1 = f1_score(y_test, ens_preds, average='macro', zero_division=0)
except Exception:
    ens_preds = None
    ens_acc = ens_prec = ens_rec = ens_f1 = 0.0

results['Ensemble'] = {
    'model': ensemble,
    'acc': ens_acc,
    'precision': ens_prec,
    'recall': ens_rec,
    'f1': ens_f1,
    'preds': ens_preds
}

print('Done')
