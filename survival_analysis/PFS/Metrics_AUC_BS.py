import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sksurv.metrics import concordance_index_censored
from sksurv.metrics import cumulative_dynamic_auc, integrated_brier_score, brier_score
import joblib
from sksurv.util import Surv
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone


model_files = [
    "CPH_none.joblib",
    "CPH_univ.joblib",
    "EST_none.joblib",
    "EST_univ.joblib",
    "SSVM_none.joblib",
    "SSVM_univ.joblib",
    "GBS_none.joblib",
    "GBS_univ.joblib",
    "RSF_none.joblib",
    "RSF_univ.joblib"
]

# Load data
X_train = pd.read_excel("Data/X_train.xlsx").iloc[:, 1:]
X_test  = pd.read_excel("Data/X_test.xlsx").iloc[:, 1:]
y_train_df = pd.read_excel("Data/y_train.xlsx").iloc[:, 1:]
y_test_df  = pd.read_excel("Data/y_test.xlsx").iloc[:, 1:]
y_train_df = y_train_df.rename(columns={"STATUS PFS": "event", "PFS": "time"})
y_test_df = y_test_df.rename(columns={"STATUS PFS": "event", "PFS": "time"})
y_train = Surv.from_dataframe("event", "time", y_train_df)
y_test  = Surv.from_dataframe("event", "time", y_test_df)

# Time points
times_obs = y_test["time"]
N = len(times_obs)
min_at_risk = int(np.ceil(0.10 * N))  # 10% threshold
candidate_times = np.sort(np.unique(times_obs))
# number at risk at each time
n_at_risk = np.array([
    np.sum(times_obs >= t)
    for t in candidate_times
])
valid_times = candidate_times[n_at_risk >= min_at_risk]
t_max = valid_times.max()
n_time_points = 20
times = np.linspace(candidate_times.min(), t_max, n_time_points)

# ==============================================================================
def cindex(y_true, y_pred):
    return concordance_index_censored(
        y_true["event"], y_true["time"], y_pred
    )[0]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=31)

def stratify_labels(y):
    return y["event"].astype(int)

class SurvivalUnivariateSelector(BaseEstimator, TransformerMixin):
    def __init__(self, model, threshold=0.55, cv=3):
        self.model = model
        self.threshold = threshold
        self.cv = cv

    def fit(self, X, y):
        Xc = X.copy()
        self.selected_features_ = []

        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)
        strat_labels = stratify_labels(y)

        for col in Xc.columns:
            scores = []

            for train_idx, val_idx in skf.split(Xc, strat_labels):
                X_tr, X_val = Xc.iloc[train_idx][[col]], Xc.iloc[val_idx][[col]]
                y_tr, y_val = y[train_idx], y[val_idx]

                model = clone(self.model)
                model.fit(X_tr, y_tr)
                preds = model.predict(X_val)

                scores.append(cindex(y_val, preds))

            if np.mean(scores) > self.threshold:
                self.selected_features_.append(col)

        if len(self.selected_features_) == 0:
            self.selected_features_ = list(Xc.columns)

        return self

    def transform(self, X):
        Xc = X.copy()
        return Xc[self.selected_features_]

# ===============================================================================

results = []

for file in model_files:
    try:
        # Extract model and fs type
        model_name = file.replace(".joblib", "")
        name, fs = model_name.rsplit("_", 1)

        # Load pipeline
        pipeline = joblib.load(os.path.join("Results", file))

        # Get selected features if any
        fs_step = pipeline.named_steps.get("fs", None)
        if hasattr(fs_step, "selected_features_"):
            features = fs_step.selected_features_
            X_tr = X_train[features]
            X_te = X_test[features]
        else:
            X_tr = X_train
            X_te = X_test

        # Get the model step (named 'model' in your pipeline)
        model = pipeline.named_steps.get("model", None)

        # Compute time-dependent AUC
        # ------------------------------------
        # Train
        pred_risks_train = model.predict(X_tr)
        time_auc_train, mean_auc_train = cumulative_dynamic_auc(y_train, y_train, pred_risks_train, times)

        # Test
        pred_risks_test = model.predict(X_te)
        time_auc_test, mean_auc_test = cumulative_dynamic_auc(y_train, y_test, pred_risks_test, times)

        plt.figure(figsize=(5.5,4))
        plt.plot(times, time_auc_test, marker="o", markersize=5)
        plt.axhline(mean_auc_test, linestyle="--")
        plt.title(f"{model_name} - time-dependent AUC")
        plt.xlabel("time (months from infusion)")
        plt.ylabel("AUC")
        plt.ylim([0, 1])
        plt.grid(True)
        plt.text(x=0.95, y=0.95, s=f"mean AUC: {mean_auc_test:.3f}", 
                va='top', ha='right', transform=plt.gca().transAxes, 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='grey'))
        
        # Save plots
        plt.savefig(f"Results/AUC_{model_name}.png")
        # ------------------------------------
        
        # Compute time-dependent Brier score
        # -------------------------------------
        
        if name != 'SSVM':
            # Train
            surv_funct_train = model.predict_survival_function(X_tr)
            sf_train = np.asarray([[fn(t) for t in times] for fn in surv_funct_train])
            _, time_brier_train = brier_score(y_train, y_train, sf_train, times)
            integ_brier_train = integrated_brier_score(y_train, y_train, sf_train, times)

            # Test
            surv_funct_test = model.predict_survival_function(X_te)
            sf_test = np.asarray([[fn(t) for t in times] for fn in surv_funct_test])
            _, time_brier_test = brier_score(y_train, y_test, sf_test, times)
            integ_brier_test = integrated_brier_score(y_train, y_test, sf_test, times)

            plt.figure(figsize=(5.5,4))
            plt.plot(times, time_brier_test, marker="o", markersize=5)
            plt.axhline(integ_brier_test, linestyle="--")
            plt.title(f"{model_name} - time-dependent Brier score")
            plt.xlabel("time (months from infusion)")
            plt.ylabel("Brier score")
            plt.ylim([0, 0.5])
            plt.grid(True)
            plt.text(x=0.05, y=0.95, s=f"integrated Brier score: {integ_brier_test:.3f}", 
                    va='top', ha='left', transform=plt.gca().transAxes, 
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='grey'))
            
            # Save plots
            plt.savefig(f"Results/BS_{model_name}.png")
        
        else:
            integ_brier_train = np.nan
            integ_brier_test = np.nan
        
        # -------------------------------------

        # Save results
        results.append({
            "model": model_name,
            "train mean_auc": mean_auc_train,
            "test mean_auc": mean_auc_test,
            "train integ_brier": integ_brier_train,
            "test integ_brier": integ_brier_test
        })
        
    except Exception as e:
        print(f"Error processing {model_name}: {e}")
        continue

# Save all results to Excel
df_results = pd.DataFrame(results)
df_results.to_excel("Results/time_variant_metrics.xlsx", index=False)