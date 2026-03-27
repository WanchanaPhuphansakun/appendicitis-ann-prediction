### 8. MODEL EVALUATION on TEST DATASET ###

from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, recall_score

# Probabilities and predictions
probs_test = model.predict(X_test, verbose=0).ravel()
preds_test = (probs_test >= 0.5).astype(int)

# Core metrics
test_auc = roc_auc_score(y_test, probs_test)
test_acc = accuracy_score(y_test, preds_test)

# Confusion matrix
cm = confusion_matrix(y_test, preds_test)
tn, fp, fn, tp = cm.ravel()

# Sensitivity and Specificity
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print(f"Accuracy: {test_acc:.4f}")
print(f"AUC (AUROC): {test_auc:.4f}")
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print("\nConfusion Matrix:")
print(cm)

### 9. SAVE MODEL and METADATA for LATER USE ###

model.save(MODEL_PATH)

meta = {
    "target_col": TARGET_COL,
    "feature_cols": feature_cols,
    "label_strategy": "1 if Diagnosis contains 'append', else 0",
}
with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2)

print("Saved model to:", MODEL_PATH)
print("Saved metadata to:", META_PATH)
