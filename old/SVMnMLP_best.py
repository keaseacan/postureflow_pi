# Mix_no_xgb.py — SVM + MLP only (no XGBoost)

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix, f1_score
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping

# === Load Data ===
df = pd.read_excel("breathing_featuresUPZ.xlsx")
# IMF_1..IMF_9
X_imf = df[[f"IMF_{i}" for i in range(1, 10)]].to_numpy()
# Person (robust-ish)
person_col = "Person" if "Person" in df.columns else ("person" if "person" in df.columns else None)
X_person = pd.get_dummies(df[person_col].astype(str), prefix="Person").to_numpy() if person_col else None
X = X_imf if X_person is None else np.concatenate([X_imf, X_person], axis=1)

le = LabelEncoder()
y = le.fit_transform(df["Position"])

# === Split train/val/test (60/20/20) ===
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, stratify=y, test_size=0.20, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, stratify=y_trainval, test_size=0.25, random_state=42
)
print(f"[info] Splits — train:{len(X_train)}  val:{len(X_val)}  test:{len(X_test)}")

# Flatten (already 2D, but keep pattern)
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat   = X_val.reshape(X_val.shape[0], -1)
X_test_flat  = X_test.reshape(X_test.shape[0], -1)

# === Class weights (balanced) ===
classes = np.unique(y_train)
cls_w = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
cw_dict = {int(c): float(w) for c, w in zip(classes, cls_w)}  # Keras class_weight

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

svm_val_pred = svm_model.predict(X_val_flat)
svm_val_acc  = accuracy_score(y_val, svm_val_pred)
svm_val_f1   = f1_score(y_val, svm_val_pred, average="macro")

svm_probs     = svm_model.predict_proba(X_test_flat)
svm_test_pred = svm_model.predict(X_test_flat)
svm_test_acc  = accuracy_score(y_test, svm_test_pred)
svm_test_f1   = f1_score(y_test, svm_test_pred, average="macro")

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
weights = (val_scores / val_scores.sum()) if val_scores.sum() > 0 else np.array([0.5, 0.5])

combined_probs = (weights[0] * svm_probs) + (weights[1] * mlp_probs)
y_pred_vote = combined_probs.argmax(axis=1)

voting_acc = accuracy_score(y_test, y_pred_vote)
voting_f1  = f1_score(y_test, y_pred_vote, average="macro")
print(f"\n🧠 F1-weighted Voting — Acc: {voting_acc:.3f} | Macro-F1: {voting_f1:.3f}")
print(f"Model Weights (by val F1): SVM={weights[0]:.3f}, MLP={weights[1]:.3f}")
print("\nClassification Report (Voting):")
print(classification_report(y_test, y_pred_vote, target_names=le.classes_, digits=4))

cm_vote = confusion_matrix(y_test, y_pred_vote)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_vote, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
plt.title("Ensemble (SVM+MLP, F1-weighted) Confusion Matrix")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.tight_layout()
plt.show()

# -----------------------------
# Stacking meta-learner (LogReg on probs)
# -----------------------------
svm_val_probs = svm_model.predict_proba(X_val_flat)
mlp_val_probs = mlp_model.predict(Xva_s, verbose=0)

Z_val  = np.hstack([svm_val_probs, mlp_val_probs])
Z_test = np.hstack([svm_probs,     mlp_probs])

meta = LogisticRegression(max_iter=2000, multi_class="multinomial")
meta.fit(Z_val, y_val)
y_pred_stack = meta.predict(Z_test)

stack_acc = accuracy_score(y_test, y_pred_stack)
stack_f1  = f1_score(y_test, y_pred_stack, average="macro")
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
