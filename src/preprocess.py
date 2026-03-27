### 1 IMPORTS & SETUP ###

import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    classification_report,
    confusion_matrix
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Set rondom seed of numpy and tensorflow to get similar result everytime.
np.random.seed(1)
tf.random.set_seed(1)

CSV_PATH   = "Regensburg Pediatric Appendicitis.csv"
TARGET_COL = "Diagnosis"                              # the column that will be used for predictions.
MODEL_PATH = "appendicitis_tf.keras"
META_PATH  = "appendicitis_tf_meta.json"


### 2. LOAD and CLEAN DATA ###

df = pd.read_csv(CSV_PATH)


# this will strip every columns in the csv
df.columns = [c.strip() for c in df.columns]

# drop any coloumns that is "unnamed"
id_like = [c for c in df.columns if c.lower().startswith("unnamed")]
df = df.drop(columns=id_like, errors="ignore")

print("Columns in dataset:")
print(df.columns.tolist())

print("\nPreview of Diagnosis column:")
print(df[TARGET_COL].head(10))


### 3. MAP DIANOSIS COLUMN to BINARY LABEL 0 or 1 ###

# This will turn appendictis to 1 and no appendictis to 0.
y_raw = df[TARGET_COL].astype(str).str.strip().str.lower()

def map_diagnosis(v: str):
    v = v.strip().lower()
    if "no append" in v:      # e.x. "no appendicitis"
        return 0
    elif "append" in v:       # e.x. "appendicitis"
        return 1
    else:
        return np.nan         # set data to NaN and drop later

y_mapped = y_raw.apply(map_diagnosis)

print("\nClass counts BEFORE dropping NaN:")
print(y_mapped.value_counts(dropna=False))

# Drop row that contains NaN.
mask_valid = y_mapped.notna()
# Show number of droped rows.
if (~mask_valid).sum() > 0:
    print("\nDropping rows with no valid label:", (~mask_valid).sum())

df = df[mask_valid].copy()
y = y_mapped[mask_valid].astype(int).to_numpy()

print("\nClass counts AFTER dropping NaN:")
print(pd.Series(y).value_counts())


### 4. BUILD FEATURES X ###

# Prepare data X to for trianing.
X = df.drop(columns=[TARGET_COL])

# check for infinite values and replace with nan.
X = X.replace([np.inf, -np.inf], np.nan)

# This line will convert catregory or text columns into numeric for the model.
X_encoded = pd.get_dummies(X, drop_first=True, dummy_na=True)

# This line will fill the NaN with the median of each column.
X_encoded = X_encoded.fillna(X_encoded.median(numeric_only=True))

feature_cols = list(X_encoded.columns)
print("Total encoded feature columns:", len(feature_cols))
print("First 15 feature columns:", feature_cols[:15])


### 5. SPLIT DATA to TRAIN/TEST ###

from sklearn.model_selection import train_test_split

X_all = X_encoded.to_numpy(dtype=np.float32)
y_all = y

def safe_train_test_split(X_all, y_all):
    try:
        return train_test_split(
            X_all, y_all,
            test_size=0.20, # split 20% of the dataset into test set.
            random_state=42,
            stratify=y_all
        )
    except ValueError:
        print("Stratified split failed, falling back to normal shuffle split.")
        return train_test_split(
            X_all, y_all,
            test_size=0.20,
            random_state=42,
            shuffle=True,
            stratify=None
        )

X_train, X_test, y_train, y_test = safe_train_test_split(X_all, y_all)

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
print("Train class balance:", np.bincount(y_train))
print("Test  class balance:", np.bincount(y_test))

