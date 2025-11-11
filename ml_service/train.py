# ml_service/train.py
import warnings, json
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score, confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

from xgboost.sklearn import XGBClassifier   # ✅ IMPORTANT — correct wrapper

# ---------- PATH SETUP ----------
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
MODELS_DIR = APP_DIR / "models"
REPORTS_DIR = APP_DIR / "reports"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

TRAIN_CSV = DATA_DIR / "Train Dataset.csv"
TEST_CSV  = DATA_DIR / "Test Dataset.csv"
TUNED_JSON = MODELS_DIR / "tuned_params.json"

FEATURES = ["N", "P", "K", "ph", "temperature", "rainfall"]
LABEL_CANDIDATES = ["label", "Label", "crop", "Crop", "target", "Target"]


# ---------- DATA HELPERS ----------
def load_csvs():
    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)
    return train_df, test_df

def pick_label_col(df):
    for col in LABEL_CANDIDATES:
        if col in df.columns:
            return col
    raise KeyError("❌ Label column not found. Ensure dataset has crop label.")


def get_best_params(model_name):
    # ...existing code...
        
    return {}


def build_models(num_classes):
    models = {
        "DecisionTree": DecisionTreeClassifier(
            random_state=42,
            **get_best_params("DecisionTree")
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=120,
            n_jobs=-1,
            random_state=42,
            **get_best_params("RandomForest")
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            n_jobs=-1,
            random_state=42,
            **get_best_params("LogisticRegression")
        ),
    }
    # Try XGBoost with GPU, fallback to CPU if not available
    try:
        xgb = XGBClassifier(
            tree_method="gpu_hist",
            predictor="gpu_predictor",
            objective="multi:softprob",
            num_class=num_classes,
            eval_metric="mlogloss",
            n_estimators=140,
            learning_rate=0.07,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            n_jobs=-1,
            random_state=42,
            **get_best_params("XGBoost")
        )
        # Test fit to check GPU availability (small dummy fit)
        import numpy as np
        xgb.fit(np.array([[0,0,0,0,0,0],[1,1,1,1,1,1]]), np.array([0,1]))
        print("✅ XGBoost GPU enabled.")
    except Exception as e:
        print("⚠️ XGBoost GPU not available, falling back to CPU. Reason:", e)
        xgb = XGBClassifier(
            tree_method="hist",
            objective="multi:softprob",
            num_class=num_classes,
            eval_metric="mlogloss",
            n_estimators=140,
            learning_rate=0.07,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            n_jobs=-1,
            random_state=42,
            **get_best_params("XGBoost")
        )
    models["XGBoost"] = xgb
    return models


def evaluate(models, X_train, y_train, X_test, y_test, encoder):
    results = {}
    detailed_metrics = {}
    for name, model in models.items():
        print(f"\n🤖 Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="macro", zero_division=0)
        rec = recall_score(y_test, preds, average="macro", zero_division=0)
        f1 = f1_score(y_test, preds, average="macro", zero_division=0)
        results[name] = {
            "model": model,
            "acc": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "preds": preds
        }
        crop_metrics = {}
        report_dict = classification_report(y_test, preds, target_names=encoder.classes_, output_dict=True, zero_division=0)
        for crop in encoder.classes_:
            if crop in report_dict:
                metrics = report_dict[crop]
                crop_metrics[crop] = {
                    "precision": round(metrics["precision"], 4),
                    "recall": round(metrics["recall"], 4),
                    "f1": round(metrics["f1-score"], 4),
                    "support": int(metrics["support"])
                }
        detailed_metrics[name] = crop_metrics
        print(f"   ✅ ACC={acc:.4f} | PREC={prec:.4f} | REC={rec:.4f} | F1={f1:.4f}")
    return results, detailed_metrics


def build_ensemble(results):
    estimators = [(name.lower(), results[name]["model"]) for name in results]
    weights = [results[name]["f1"] for name in results]   # weighted soft voting
    return VotingClassifier(estimators=estimators, voting="soft", weights=weights)


# ---------- GENERATE DETAILED REPORTS ----------
def generate_reports(results, detailed_metrics, encoder, X_test_s, y_test, ensemble_preds):
    """Generate CSV, JSON, and text reports with detailed metrics."""
    
    # 1. Build comparison dataframe
    crops = list(encoder.classes_)
    comparison_data = []
    
    for crop in crops:
        row = {"Crop": crop}
        
        # Get support (number of samples for this crop)
        support = sum(y_test == encoder.transform([crop])[0])
        row["Support"] = support
        
        # Add metrics for each model
        for model_name in results.keys():
            if crop in detailed_metrics[model_name]:
                metrics = detailed_metrics[model_name][crop]
                row[f"{model_name}_Precision"] = metrics["precision"]
                row[f"{model_name}_Recall"] = metrics["recall"]
                row[f"{model_name}_F1"] = metrics["f1"]
        
        # Add ensemble metrics (calculate from ensemble predictions)
        from sklearn.metrics import precision_score, recall_score, f1_score
        crop_idx = encoder.transform([crop])[0]
        # Compute per-crop binary metrics for ensemble (one-vs-rest)
        y_true_binary = (y_test == crop_idx).astype(int)
        y_pred_binary = (ensemble_preds == crop_idx).astype(int)
        support_count = int(y_true_binary.sum())
        if support_count > 0:
            ens_prec = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            ens_rec = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            ens_f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
            row["Ensemble_Precision"] = round(ens_prec, 4)
            row["Ensemble_Recall"] = round(ens_rec, 4)
            row["Ensemble_F1"] = round(ens_f1, 4)
            row["Support"] = support_count
        
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # 2. Save CSV
    csv_path = REPORTS_DIR / "detailed_metrics.csv"
    comparison_df.to_csv(csv_path, index=False)
    print(f"\n💾 Saved detailed metrics to: {csv_path}")
    
    # 3. Save JSON
    json_path = REPORTS_DIR / "detailed_metrics.json"
    json_data = {
        "crops": crops,
        "detailed_metrics": detailed_metrics,
        "comparison": comparison_df.to_dict(orient="records")
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"💾 Saved JSON metrics to: {json_path}")
    
    return comparison_df

    # 4. Confusion matrix for ensemble
    try:
        cm = confusion_matrix(y_test, ensemble_preds)
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, xticklabels=crops, yticklabels=crops, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix - Ensemble")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        cm_path = REPORTS_DIR / "confusion_matrix_ensemble.png"
        plt.savefig(cm_path, dpi=200)
        plt.close()
        print(f"💾 Saved ensemble confusion matrix to: {cm_path}")
    except Exception as e:
        print("⚠️ Could not generate confusion matrix plot:", e)
    

    
    return comparison_df


def print_detailed_report(results, detailed_metrics, comparison_df, encoder):
    """Print model performance summary."""
    
    print("\n" + "="*90)
    print("📈 MODEL PERFORMANCE SUMMARY")
    print("="*90)
    print(f"{'Model':<20} {'Accuracy':>12} {'Precision':>12} {'Recall':>12} {'F1-Score':>12}")
    print("-" * 90)
    
    for name, result in results.items():
        print(f"{name:<20} {result['acc']:>12.4f} {result['precision']:>12.4f} {result['recall']:>12.4f} {result['f1']:>12.4f}")


def print_model_parameters(results):
    """Print hyperparameters for each model."""
    
    print("\n" + "="*90)
    print("⚙️  MODEL HYPERPARAMETERS")
    print("="*90)
    
    model_names = list(results.keys())
    
    for name in model_names:
        model = results[name]["model"]
        print(f"\n🔧 {name}:")
        print("-" * 90)
        
        if name == "DecisionTree":
            print(f"   random_state: 42")
            print(f"   max_depth: {model.get_params().get('max_depth', 'unlimited')}")
            print(f"   min_samples_split: {model.get_params().get('min_samples_split', 2)}")
            print(f"   min_samples_leaf: {model.get_params().get('min_samples_leaf', 1)}")
        
        elif name == "RandomForest":
            print(f"   n_estimators: 120 (120 trees)")
            print(f"   max_depth: {model.get_params().get('max_depth', 'unlimited')}")
            print(f"   min_samples_split: {model.get_params().get('min_samples_split', 2)}")
            print(f"   n_jobs: -1 (all CPUs)")
            print(f"   random_state: 42")
        
        elif name == "LogisticRegression":
            print(f"   max_iter: 2000 (max iterations)")
            print(f"   solver: {model.get_params().get('solver', 'lbfgs')}")
            print(f"   C: {model.get_params().get('C', 1.0)} (inverse regularization)")
            print(f"   n_jobs: -1 (all CPUs)")
            print(f"   random_state: 42")
        
        elif name == "XGBoost":
            print(f"   tree_method: gpu_hist (GPU acceleration)")
            print(f"   predictor: gpu_predictor (GPU inference)")
            print(f"   objective: multi:softprob (multi-class soft probabilities)")
            print(f"   n_estimators: 140 (140 boosting rounds)")
            print(f"   learning_rate: 0.07 (step shrinkage)")
            print(f"   max_depth: 6 (max tree depth)")
            print(f"   subsample: 0.9 (90% of samples per tree)")
            print(f"   colsample_bytree: 0.9 (90% of features per tree)")
            print(f"   random_state: 42")


# ---------- MAIN ----------
def generate_model_comparison_charts(results, reports_dir):
    """Generate overall model comparison charts (grouped bar) and ensemble weights."""
    try:
        # Build DataFrame of metrics
        rows = []
        for name, res in results.items():
            rows.append({
                "Model": name,
                "Accuracy": round(res.get("acc", 0), 4),
                "Precision": round(res.get("precision", 0), 4),
                "Recall": round(res.get("recall", 0), 4),
                "F1": round(res.get("f1", 0), 4),
            })

        comp_df = pd.DataFrame(rows)
        comp_csv = reports_dir / "model_comparison_metrics.csv"
        comp_df.to_csv(comp_csv, index=False)

        # Grouped bar chart
        plt.figure(figsize=(10, 6))
        mdf = comp_df.melt(id_vars=["Model"], value_vars=["Accuracy", "Precision", "Recall", "F1"], var_name="Metric", value_name="Score")
        sns.barplot(data=mdf, x="Model", y="Score", hue="Metric")
        plt.ylim(0, 1)
        plt.xticks(rotation=30)
        plt.title("Model comparison — Accuracy / Precision / Recall / F1")
        plt.tight_layout()
        gb_path = reports_dir / "model_comparison_metrics.png"
        plt.savefig(gb_path, dpi=200)
        plt.close()
        print(f"💾 Saved model comparison grouped-bar to: {gb_path}")

        # Ensemble weights bar (if ensemble present)
        if "Ensemble" in results:
            try:
                weights = getattr(results["Ensemble"]["model"], "weights", None)
                if weights is None:
                    weights = results["Ensemble"]["model"].weights
                model_names = [m for m in results.keys() if m != "Ensemble"]
                w_df = pd.DataFrame({"Model": model_names, "Weight": list(weights)})
                plt.figure(figsize=(8, 4))
                sns.barplot(data=w_df, x="Model", y="Weight", palette="crest")
                plt.title("Ensemble Weights (soft voting)")
                plt.xticks(rotation=30)
                plt.tight_layout()
                w_path = reports_dir / "ensemble_weights.png"
                plt.savefig(w_path, dpi=200)
                plt.close()
                print(f"💾 Saved ensemble weights chart to: {w_path}")
            except Exception as e:
                print("⚠️ Could not generate ensemble weights chart:", e)

    except Exception as e:
        print("⚠️ Could not generate model comparison charts:", e)


def generate_model_parameters_comparison(results, reports_dir):
    """Generate a comparison chart showing all model parameters."""
    try:
        # Extract parameters for each model
        params_data = []
        
        for model_name in results.keys():
            model = results[model_name]["model"]
            params = model.get_params()
            
            # Build parameter info string
            if model_name == "DecisionTree":
                param_str = f"""DecisionTree
random_state: {params.get('random_state', 'N/A')}
max_depth: {params.get('max_depth', 'None')}
min_samples_split: {params.get('min_samples_split', 2)}
min_samples_leaf: {params.get('min_samples_leaf', 1)}"""
            
            elif model_name == "RandomForest":
                param_str = f"""RandomForest
n_estimators: 120
max_depth: {params.get('max_depth', 'None')}
min_samples_split: {params.get('min_samples_split', 2)}
n_jobs: -1
random_state: {params.get('random_state', 'N/A')}"""
            
            elif model_name == "LogisticRegression":
                param_str = f"""LogisticRegression
max_iter: {params.get('max_iter', 2000)}
solver: {params.get('solver', 'lbfgs')}
C: {params.get('C', 1.0)}
n_jobs: -1
random_state: {params.get('random_state', 'N/A')}"""
            
            elif model_name == "XGBoost":
                param_str = f"""XGBoost
n_estimators: 140
learning_rate: 0.07
max_depth: 6
subsample: 0.9
colsample_bytree: 0.9
tree_method: gpu_hist
random_state: {params.get('random_state', 'N/A')}"""
            
            elif model_name == "Ensemble":
                param_str = f"""Ensemble (Soft Voting)
voting: soft
weights: F1-based
DecisionTree weight: {results[model_name]['model'].weights[0]:.4f}
RandomForest weight: {results[model_name]['model'].weights[1]:.4f}
LogisticRegression weight: {results[model_name]['model'].weights[2]:.4f}
XGBoost weight: {results[model_name]['model'].weights[3]:.4f}"""
            
            params_data.append({
                "Model": model_name,
                "Parameters": param_str
            })
        
        # Create figure with text-based comparison
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('off')
        
        y_pos = 0.95
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFD700', '#FF99CC']
        
        for idx, item in enumerate(params_data):
            model_name = item["Model"]
            param_str = item["Parameters"]
            
            # Add background box
            bbox = dict(boxstyle="round,pad=0.5", facecolor=colors[idx], alpha=0.3, edgecolor='black', linewidth=1.5)
            ax.text(0.5, y_pos, param_str, ha='center', va='top', fontsize=10, 
                   family='monospace', bbox=bbox, transform=ax.transAxes)
            
            y_pos -= 0.19
        
        plt.title("Model Parameters Comparison", fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        params_path = reports_dir / "model_parameters_comparison.png"
        plt.savefig(params_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved model parameters comparison to: {params_path}")
    
    except Exception as e:
        print("⚠️ Could not generate model parameters comparison:", e)







def main():
    print("\n🌱 Loading dataset...")
    train_df, test_df = load_csvs()

    label = pick_label_col(train_df)

    X_train = train_df[FEATURES].astype(float).values
    X_test  = test_df[FEATURES].astype(float).values
    y_train_raw = train_df[label].astype(str).values
    y_test_raw  = test_df[label].astype(str).values

    print("⚙️ Scaling & encoding...")
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)

    encoder = LabelEncoder().fit(y_train_raw)
    y_train = encoder.transform(y_train_raw)
    y_test  = encoder.transform(y_test_raw)

    num_classes = len(encoder.classes_)

    print("\n🤖 Training Base Models...\n")
    models = build_models(num_classes)
    results, detailed_metrics = evaluate(models, X_train_s, y_train, X_test_s, y_test, encoder)

    # Print hyperparameters
    print_model_parameters(results)

    print("\n🧠 Building Weighted Soft Voting Ensemble...")
    ensemble = build_ensemble(results)
    ensemble.fit(X_train_s, y_train)

    ensemble_preds = ensemble.predict(X_test_s)
    ensemble_acc = accuracy_score(y_test, ensemble_preds)
    ensemble_prec = precision_score(y_test, ensemble_preds, average="macro", zero_division=0)
    ensemble_rec = recall_score(y_test, ensemble_preds, average="macro", zero_division=0)
    ensemble_f1 = f1_score(y_test, ensemble_preds, average="macro", zero_division=0)

    print(f"\n✅ ENSEMBLE ACCURACY:  {ensemble_acc:.4f}")
    print(f"✅ ENSEMBLE PRECISION: {ensemble_prec:.4f}")
    print(f"✅ ENSEMBLE RECALL:    {ensemble_rec:.4f}")
    print(f"✅ ENSEMBLE F1-SCORE:  {ensemble_f1:.4f}\n")

    # Add ensemble to results and detailed metrics
    from sklearn.metrics import classification_report
    ensemble_report = classification_report(y_test, ensemble_preds, target_names=encoder.classes_, output_dict=True, zero_division=0)
    ensemble_metrics = {}
    for crop in encoder.classes_:
        if crop in ensemble_report:
            metrics = ensemble_report[crop]
            ensemble_metrics[crop] = {
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "f1": round(metrics["f1-score"], 4),
                "support": int(metrics["support"])
            }
    
    results["Ensemble"] = {
        "model": ensemble,
        "acc": ensemble_acc,
        "precision": ensemble_prec,
        "recall": ensemble_rec,
        "f1": ensemble_f1,
        "preds": ensemble_preds
    }
    detailed_metrics["Ensemble"] = ensemble_metrics

    # Generate detailed reports
    comparison_df = generate_reports(results, detailed_metrics, encoder, X_test_s, y_test, ensemble_preds)

    # Print detailed report
    print_detailed_report(results, detailed_metrics, comparison_df, encoder)

    # Print classification report
    print("\n📊 ENSEMBLE CLASSIFICATION REPORT:\n")
    print(classification_report(y_test, ensemble_preds, target_names=encoder.classes_))

    print("\n💾 Saving Model Files...")
    joblib.dump(ensemble, MODELS_DIR / "ensemble_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(encoder, MODELS_DIR / "label_encoder.pkl")

    # Save comprehensive training report
    training_report = {
        "base_models": {
            name: {
                "accuracy": round(results[name]["acc"], 4),
                "precision": round(results[name]["precision"], 4),
                "recall": round(results[name]["recall"], 4),
                "f1": round(results[name]["f1"], 4)
            }
            for name in list(results.keys())[:-1]  # Exclude ensemble here
        },
        "ensemble": {
            "accuracy": round(ensemble_acc, 4),
            "precision": round(ensemble_prec, 4),
            "recall": round(ensemble_rec, 4),
            "f1": round(ensemble_f1, 4),
            "weights": [round(w, 4) for w in ensemble.weights]
        },
        "features": FEATURES,
        "label_classes": list(encoder.classes_)
    }
    
    with open(MODELS_DIR / "training_report.json", "w") as f:
        json.dump(training_report, f, indent=2)

    print("\n✅ DONE — Model Successfully Trained & Saved.\n")
    print(f"📁 Reports saved to: {REPORTS_DIR}/\n")

    # Generate overall model comparison charts (grouped bar, weights)
    try:
        generate_model_comparison_charts(results, REPORTS_DIR)
    except Exception as e:
        print("⚠️ Could not generate overall comparison charts:", e)
    
    # Generate model parameters comparison
    try:
        generate_model_parameters_comparison(results, REPORTS_DIR)
    except Exception as e:
        print("⚠️ Could not generate model parameters comparison:", e)





if __name__ == "__main__":
    main()
