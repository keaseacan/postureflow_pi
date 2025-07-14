import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, LeakyReLU
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns

# === Load Data ===
df = pd.read_excel("breathing_features.xlsx")
X_imf = df[[f"IMF_{i}" for i in range(1, 10)]].to_numpy()
X_person = pd.get_dummies(df["Person"], prefix="Person").to_numpy()
X = np.concatenate([X_imf, X_person], axis=1)
le = LabelEncoder()
y = le.fit_transform(df["Position"])

# === Split train/val/test ===
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, stratify=y_trainval, test_size=0.25, random_state=42)

# === Flatten ===
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# === MLP ===
mlp_model = Sequential([
    Input(shape=(X_train_flat.shape[1],)),
    Dense(64), LeakyReLU(negative_slope=0.1), Dropout(0.2),
    Dense(64), LeakyReLU(negative_slope=0.1), Dropout(0.2),
    Dense(len(le.classes_), activation='softmax')
])
mlp_model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
mlp_model.fit(X_train_flat, y_train, validation_data=(X_val_flat, y_val), epochs=60, batch_size=16, verbose=0)
mlp_val_acc = accuracy_score(y_val, np.argmax(mlp_model.predict(X_val_flat), axis=1))
mlp_probs = mlp_model.predict(X_test_flat)

# === XGBoost ===
xgb_model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=len(le.classes_),
    eval_metric='mlogloss',
    n_estimators=300,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train_flat, y_train)
xgb_val_acc = accuracy_score(y_val, np.argmax(xgb_model.predict_proba(X_val_flat), axis=1))
xgb_probs = xgb_model.predict_proba(X_test_flat)

# === SVM ===
svm_model = make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True))
svm_model.fit(X_train_flat, y_train)
svm_val_acc = accuracy_score(y_val, svm_model.predict(X_val_flat))
svm_probs = svm_model.predict_proba(X_test_flat)

# === Auto-weighted Voting ===
accs = np.array([svm_val_acc, xgb_val_acc, mlp_val_acc])
weights = accs / accs.sum()

combined_probs = (weights[0] * svm_probs) + (weights[1] * xgb_probs) + (weights[2] * mlp_probs)
y_pred_vote = np.argmax(combined_probs, axis=1)

# === Results ===
voting_acc = accuracy_score(y_test, y_pred_vote)
print(f"🧠 Auto-Weighted Voting Accuracy: {voting_acc:.2f}")
print(f"Model Weights: SVM={weights[0]:.2f}, XGB={weights[1]:.2f}, MLP={weights[2]:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_vote, target_names=le.classes_))

cm = confusion_matrix(y_test, y_pred_vote)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
plt.title("Auto-Weighted Voting Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()
