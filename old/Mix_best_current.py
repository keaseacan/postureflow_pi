#!/usr/bin/env python3
# Mix_no_xgb_calibrated.py — SVM + MLP (no XGB), calibrated, F1-weighted + optimized voting, stacking, auto-pick final

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping

# === Load Data ===
df = pd.read_excel("breathing_featuresUPZ.xlsx")

# IMF_1..IMF_9 (adjust if needed)
imf_cols = [f"IMF_{i}" for i in range(1, 10)]
X_imf = df[imf_cols].to_numpy()

# Person one-hot (robust to case / optional)
person_col = "Person" if "Person" in df.columns else ("person" if "person" in df.columns else None)
X_person = pd.get_dummies(df[person_col].astype(str), prefix="Person").to_numpy() if person_col else None

X = X_imf if X_person is None else np.concatenate([X_imf, X_person], axis=1)

# Labels
le = LabelEncoder()
y = le.fit_transform(df["Position"].astype(str).values)

# === Helper: flexible train/val/test split ===
def split_train_val_test(X, y, test=0.20, val=0.20, seed=42, stratify=True):
    """Return X_train, X_val, X_test, y_train, y_val, y_test with exact fractions."""
    assert 0 < test < 1 and 0 < val < 1 and test + val < 1, "Fractions must sum to < 1"
    y_strat = y if stratify else None
    X_tv, X_te, y_tv, y_te = train_test_split(X, y, test_size=test, stratify=y_strat, random_state=seed)
    val_rel = val / (1.0 - test)
    y_tv_strat = y_tv if stratify else None
    X_tr, X_va, y_tr, y_va = train_test_split(X_tv, y_tv, test_size=val_rel, stratify=y_tv_strat, random_state=seed)
    return X_tr, X_va, X_te, y_tr, y_va, y_te

# === Choose your split fractions here ===
TEST = 0.20  # final test fraction
VAL  = 0.20  # final validation fraction

# Do the split
X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
    X, y, test=TEST, val=VAL, seed=42, stratify=True
)
print(f"[info] Splits — train:{len(X_train)}  val:{len(X_val)}  test:{len(X_test)}")

# Flatten (already 2D, but keep pattern)
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat   = X_val.reshape(X_val.shape[0], -1)
X_test_flat  = X_test.reshape(X_test.shape[0], -1)

# === Class weights (balanced) ===
classes = np.unique(y_train)
cls_w = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
cw_dict = {int(c): float(w) for c, w in zip(classes, cls_w)}  # for Keras

# -----------------------------
# SVM — grid search on macro-F1
# -----------------------------
svm_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)),
])
param_grid = {
    "svc__C":     [0.5, 1, 2, 4, 8, 16],
    "svc__gamma": ["scale", 0.2, 0.1, 0.05, 0.02, 0.01],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
svm_search = GridSearchCV(svm_pipe, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1, verbose=0)
svm_search.fit(X_train_flat, y_train)
svm_model = svm_search.best_estimator_
print(f"[svm] best params: {svm_search.best_params_}  cv_f1={svm_search.best_score_:.3f}")

# Raw SVM pipeline metrics (pre-calibration)
svm_val_pred = svm_model.predict(X_val_flat)
svm_val_acc  = accuracy_score(y_val, svm_val_pred)
svm_val_f1   = f1_score(y_val, svm_val_pred, average="macro")
svm_test_pred = svm_model.predict(X_test_flat)
svm_test_acc  = accuracy_score(y_test, svm_test_pred)
svm_test_f1   = f1_score(y_test, svm_test_pred, average="macro")

# --- Calibrate SVM probabilities with CV on TRAIN (no deprecation, no leakage) ---
svm_cal = CalibratedClassifierCV(svm_model, method="sigmoid", cv=5)
svm_cal.fit(X_train_flat, y_train)
# calibrated probs (pipeline handles scaling internally)
svm_val_probs = svm_cal.predict_proba(X_val_flat)
svm_probs     = svm_cal.predict_proba(X_test_flat)

# -----------------------------
# MLP — scaling + L2 + dropout + early stopping + class weights
# -----------------------------
scaler_mlp = StandardScaler().fit(X_train_flat)
Xtr_s = scaler_mlp.transform(X_train_flat)
Xva_s = scaler_mlp.transform(X_val_flat)
Xte_s = scaler_mlp.transform(X_test_flat)

mlp_model = Sequential([
    Input(shape=(Xtr_s.shape[1],)),
    Dense(128, kernel_regularizer=regularizers.l2(1e-4)), LeakyReLU(0.1), Dropout(0.3),
    Dense(64,  kernel_regularizer=regularizers.l2(1e-4)), LeakyReLU(0.1), Dropout(0.2),
    Dense(len(le.classes_), activation="softmax"),
])
mlp_model.compile(optimizer=Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
es = EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, min_delta=1e-4)
mlp_model.fit(Xtr_s, y_train, validation_data=(Xva_s, y_val),
              epochs=200, batch_size=32, verbose=0, callbacks=[es], class_weight=cw_dict)

mlp_val_pred = mlp_model.predict(Xva_s, verbose=0).argmax(1)
mlp_val_acc  = accuracy_score(y_val, mlp_val_pred)
mlp_val_f1   = f1_score(y_val, mlp_val_pred, average="macro")

mlp_probs     = mlp_model.predict(Xte_s, verbose=0)
mlp_test_pred = mlp_probs.argmax(1)
mlp_test_acc  = accuracy_score(y_test, mlp_test_pred)
mlp_test_f1   = f1_score(y_test, mlp_test_pred, average="macro")

# -----------------------------
# Summary per-model
# -----------------------------
print("\nIndividual model (val acc/f1 → test acc/f1):")
print(f"  SVM : {svm_val_acc:.3f}/{svm_val_f1:.3f} → {svm_test_acc:.3f}/{svm_test_f1:.3f}")
print(f"  MLP : {mlp_val_acc:.3f}/{mlp_val_f1:.3f} → {mlp_test_acc:.3f}/{mlp_test_f1:.3f}")

# -----------------------------
# F1-weighted voting (SVM + MLP)
# -----------------------------
val_scores = np.array([svm_val_f1, mlp_val_f1], dtype=float)
weights_f1 = (val_scores / val_scores.sum()) if val_scores.sum() > 0 else np.array([0.5, 0.5])

mlp_val_probs = mlp_model.predict(Xva_s, verbose=0)
vote_val_probs_f1w = weights_f1[0]*svm_val_probs + weights_f1[1]*mlp_val_probs
vote_val_pred_f1w  = vote_val_probs_f1w.argmax(1)
vote_val_f1_f1w    = f1_score(y_val, vote_val_pred_f1w, average="macro")

combined_probs_f1w = (weights_f1[0] * svm_probs) + (weights_f1[1] * mlp_probs)
y_pred_vote_f1w    = combined_probs_f1w.argmax(axis=1)
voting_acc_f1w     = accuracy_score(y_test, y_pred_vote_f1w)
voting_f1_f1w      = f1_score(y_test, y_pred_vote_f1w, average="macro")

print(f"\n🧠 F1-weighted Voting — Acc: {voting_acc_f1w:.3f} | Macro-F1: {voting_f1_f1w:.3f}")
print(f"Model Weights (by val F1): SVM={weights_f1[0]:.3f}, MLP={weights_f1[1]:.3f}")
print("\nClassification Report (Voting F1-weighted):")
print(classification_report(y_test, y_pred_vote_f1w, target_names=le.classes_, digits=4))

cm_vote_f1w = confusion_matrix(y_test, y_pred_vote_f1w)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_vote_f1w, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
plt.title("Ensemble (SVM+MLP) — F1-weighted Confusion Matrix")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.tight_layout()
plt.show()

# -----------------------------
# Validation-optimized voting weight sweep (SVM weight w ∈ [0,1])
# -----------------------------
ws = np.linspace(0.0, 1.0, 101)
best = (-1.0, None)  # (f1, w)
for w in ws:
    probs = w * svm_val_probs + (1.0 - w) * mlp_val_probs
    f1v = f1_score(y_val, probs.argmax(1), average="macro")
    if f1v > best[0]:
        best = (f1v, w)

best_w = best[1]
print(f"[vote] best_w (SVM weight) on VAL: {best_w:.2f}  val_macroF1={best[0]:.3f}")

combined_probs_opt = best_w * svm_probs + (1.0 - best_w) * mlp_probs
y_pred_vote_opt    = combined_probs_opt.argmax(axis=1)
voting_acc_opt     = accuracy_score(y_test, y_pred_vote_opt)
voting_f1_opt      = f1_score(y_test, y_pred_vote_opt, average="macro")
print(f"\n🧠 F1-optimized Voting — Acc: {voting_acc_opt:.3f} | Macro-F1: {voting_f1_opt:.3f}")
print(f"Model Weights (opt): SVM={best_w:.3f}, MLP={(1.0-best_w):.3f}")

cm_vote_opt = confusion_matrix(y_test, y_pred_vote_opt)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_vote_opt, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Oranges")
plt.title("Ensemble (SVM+MLP) — F1-optimized Confusion Matrix")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.tight_layout()
plt.show()

# -----------------------------
# Stacking meta-learner (LogReg)
# -----------------------------
Z_val  = np.hstack([svm_val_probs, mlp_val_probs])
Z_test = np.hstack([svm_probs,     mlp_probs])

meta = LogisticRegression(max_iter=2000, multi_class="multinomial")
meta.fit(Z_val, y_val)
y_pred_stack   = meta.predict(Z_test)
stack_acc      = accuracy_score(y_test, y_pred_stack)
stack_f1       = f1_score(y_test, y_pred_stack, average="macro")
stack_val_pred = meta.predict(Z_val)
stack_val_f1   = f1_score(y_val, stack_val_pred, average="macro")

print(f"\n🔗 Stacking — Acc: {stack_acc:.3f} | Macro-F1: {stack_f1:.3f}")
print("Classification Report (Stacking):")
print(classification_report(y_test, y_pred_stack, target_names=le.classes_, digits=4))

cm_stack = confusion_matrix(y_test, y_pred_stack)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_stack, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Greens")
plt.title("Stacking (SVM+MLP) Confusion Matrix")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.tight_layout()
plt.show()

# -----------------------------
# Auto-pick final model by validation macro-F1
# -----------------------------
# Use validation macro-F1 for each candidate
val_score_map = {
    "SVM":   svm_val_f1,
    "MLP":   mlp_val_f1,
    "VoteF1": vote_val_f1_f1w,   # F1-weighted vote (val)
    "VoteOpt": best[0],          # optimized vote (val)
    "Stack":  stack_val_f1
}
final_choice = max(val_score_map, key=val_score_map.get)

# Map choice to TEST predictions/metrics
if final_choice == "SVM":
    final_pred, final_acc, final_f1 = svm_test_pred, svm_test_acc, svm_test_f1
elif final_choice == "MLP":
    final_pred, final_acc, final_f1 = mlp_test_pred, mlp_test_acc, mlp_test_f1
elif final_choice == "Stack":
    final_pred, final_acc, final_f1 = y_pred_stack, stack_acc, stack_f1
elif final_choice == "VoteOpt":
    final_pred, final_acc, final_f1 = y_pred_vote_opt, voting_acc_opt, voting_f1_opt
else:  # "VoteF1"
    final_pred, final_acc, final_f1 = y_pred_vote_f1w, voting_acc_f1w, voting_f1_f1w

print(f"\n🏁 Final (chosen by val macro-F1): {final_choice} — Acc: {final_acc:.3f} | Macro-F1: {final_f1:.3f}")
print("Classification Report (Final):")
print(classification_report(y_test, final_pred, target_names=le.classes_, digits=4))

cm_final = confusion_matrix(y_test, final_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_final, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Purples")
plt.title(f"Final Model: {final_choice} — Confusion Matrix")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.tight_layout()
plt.show()
