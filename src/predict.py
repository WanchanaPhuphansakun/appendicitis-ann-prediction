### 10. RELOAD MODEL and TEST on DATASET ###

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix
)
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

np.random.seed(1)
tf.random.set_seed(1)

CSV_PATH   = "Regensburg Pediatric Appendicitis.csv"
TARGET_COL = "Diagnosis"
MODEL_PATH = "appendicitis_tf.keras"
META_PATH  = "appendicitis_tf_meta.json"


loaded_model = tf.keras.models.load_model(MODEL_PATH)

with open(META_PATH, "r") as f:
    meta_loaded = json.load(f)

loaded_feature_cols = meta_loaded["feature_cols"]
print("Loaded feature count:", len(loaded_feature_cols))


df_check = pd.read_csv(CSV_PATH)
df_check.columns = [c.strip() for c in df_check.columns]


df_check = df_check.drop(
    columns=[c for c in df_check.columns if c.lower().startswith("unnamed")],
    errors="ignore"
)


TARGET_COL = "Diagnosis"

def map_diagnosis_reload(v: str):
    v = str(v).strip().lower()
    if "no append" in v:
        return 0
    elif "append" in v:
        return 1
    else:
        return np.nan

y_check_series = df_check[TARGET_COL].apply(map_diagnosis_reload)

# Drop rows that can't be label
mask_valid2 = y_check_series.notna()
if (~mask_valid2).sum() > 0:
    print("Dropping rows with unmapped Diagnosis during reload:",
          (~mask_valid2).sum())

df_check   = df_check[mask_valid2].copy()
y_check_np = y_check_series[mask_valid2].astype(int).to_numpy()


if len(np.unique(y_check_np)) < 2:
    raise ValueError(
        "After reload preprocessing, only one class is present. "
        "Cannot verify binary performance."
    )


X_check = df_check.drop(columns=[TARGET_COL])


X_check = X_check.replace([np.inf, -np.inf], np.nan)


X_check = pd.get_dummies(X_check, drop_first=True, dummy_na=True)

# Fill missing column with medians
X_check = X_check.fillna(X_check.median(numeric_only=True))


for col in loaded_feature_cols:
    if col not in X_check.columns:
        X_check[col] = 0.0


X_check = X_check[loaded_feature_cols]


X_check_np = X_check.to_numpy(dtype=np.float32)


X_train_tmp, X_test_tmp, y_train_tmp, y_test_tmp = train_test_split(
    X_check_np,
    y_check_np,
    test_size=0.20,
    random_state=42,
    stratify=y_check_np if len(np.unique(y_check_np)) > 1 else None
)

# Evaluate the reloaded model on the recreated test split
probs_loaded = loaded_model.predict(X_test_tmp, verbose=0).ravel()
preds_loaded = (probs_loaded >= 0.5).astype(int)

# AUROC (AUC)
auc_loaded = roc_auc_score(y_test_tmp, probs_loaded)

# Accuracy
acc_loaded = accuracy_score(y_test_tmp, preds_loaded)

# Confusion matrix to get TP, TN, FP, FN
cm = confusion_matrix(y_test_tmp, preds_loaded)
tn, fp, fn, tp = cm.ravel()

sensitivity_loaded = tp / (tp + fn) if (tp + fn) > 0 else np.nan

specificity_loaded = tn / (tn + fp) if (tn + fp) > 0 else np.nan


print("========================================")
print("RELOADED MODEL METRICS (TEST SET)")
print("----------------------------------------")
print(f"Accuracy     : {acc_loaded:.4f}")
print(f"AUROC (AUC)  : {auc_loaded:.4f}")
print(f"Sensitivity  : {sensitivity_loaded:.4f}")
print(f"Specificity  : {specificity_loaded:.4f}")
print("----------------------------------------")
