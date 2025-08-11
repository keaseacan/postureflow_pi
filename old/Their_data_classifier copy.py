import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns

# === Load and preprocess data ===
df = pd.read_excel("breathing_features.xlsx")
df["Person"] = df["Person"].astype(str).str.strip()
imf_cols = [f"IMF_{i}" for i in range(1, 10)]
df = df.dropna(subset=imf_cols + ["Person", "Position"])

# === Features and labels ===
X_imf = df[imf_cols].to_numpy()
X_person = pd.get_dummies(df["Person"], prefix="Person").to_numpy()
X = np.concatenate([X_imf, X_person], axis=1)
le = LabelEncoder()
y = le.fit_transform(df["Position"])

# === Optional: Group-aware split (by speaker) ===
groups = df["Person"]
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))
X_trainval, X_test = X[train_idx], X[test_idx]
y_trainval, y_test = y[train_idx], y[test_idx]

# === Train/val split ===
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, stratify=y_trainval, test_size=0.25, random_state=42
)

# === MLP with regularization ===
mlp_model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64), LeakyReLU(0.1), Dropout(0.4),
    Dense(32), LeakyReLU(0.1), Dropout(0.3),
    Dense(len(le.classes_), activation='softmax')
])
mlp_model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
early_stop = EarlyStopping(patience=5, restore_best_weights=True)
mlp_model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=30,
              batch_size=16,
              callbacks=[early_stop],
              verbose=0)
mlp_val_acc = accuracy_score(y_val, np.argmax(mlp_model.predict(X_val), axis=1))
mlp_probs = mlp_model.predict(X_test)

# === XGBoost ===
xgb_model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=len(le.classes_),
    eval_metric='mlogloss',
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.7,
    random_state=42
)
xgb_model.fit(X_train, y_train)
xgb_val_acc = accuracy_score(y_val, np.argmax(xgb_model.predict_proba(X_val), axis=1))
xgb_probs = xgb_model.predict_proba(X_test)

# === SVM with pipeline ===
svm_model = make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True))
svm_model.fit(X_train, y_train)
svm_val_acc = accuracy_score(y_val, svm_model.predict(X_val))
svm_probs = svm_model.predict_proba(X_test)

# === Auto-weighted voting ===
accs = np.array([svm_val_acc, xgb_val_acc, mlp_val_acc])
weights = accs / accs.sum()
combined_probs = (weights[0] * svm_probs) + (weights[1] * xgb_probs) + (weights[2] * mlp_probs)
y_pred_vote = np.argmax(combined_probs, axis=1)

# === Evaluation ===
voting_acc = accuracy_score(y_test, y_pred_vote)
print(f"🧠 Auto-Weighted Voting Accuracy: {voting_acc:.2f}")
print(f"Model Weights: SVM={weights[0]:.2f}, XGB={weights[1]:.2f}, MLP={weights[2]:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_vote, target_names=le.classes_))

cm = confusion_matrix(y_test, y_pred_vote)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
plt.title("Confusion Matrix - Auto-Regularized Ensemble")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()
