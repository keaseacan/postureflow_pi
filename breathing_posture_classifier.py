
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GRU, Dense, Dropout

# === 1. Load and group fixed-length signals from CSV ===
df = pd.read_csv("breathing_data.csv")  # Replace with your file path
signals = []
labels = []

for signal_id, group in df.groupby("signal_id"):
    signals.append(group["value"].to_numpy())
    labels.append(group["label"].iloc[0])

X = np.stack(signals)  # shape: (samples, timesteps)
y = LabelEncoder().fit_transform(labels)

# === 2. Reshape for Conv1D input ===
X = X[..., np.newaxis]  # shape: (samples, timesteps, 1)

# === 3. Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# === 4. CNN-GRU Model ===
model = Sequential([
    Conv1D(64, 5, activation='relu', input_shape=(X.shape[1], 1)),
    Dropout(0.3),
    Conv1D(128, 3, activation='relu'),
    Dropout(0.3),
    GRU(64, return_sequences=False, dropout=0.3),
    Dense(len(np.unique(y)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# === 5. Train and evaluate ===
model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=16)
loss, acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {acc:.2f}")


from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# === 6. Detailed evaluation report ===
y_pred = model.predict(X_test).argmax(axis=1)

# Print formatted table
acc = np.mean(y_pred == y_test)
print("\\n=== Evaluation Summary ===")
print(f"{'Model':<15} | {'Test Accuracy'}")
print("-" * 30)
print(f"{'CNN-GRU':<15} | {acc:.2f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\\nConfusion Matrix:")
print(cm)

# Optional: Visualize
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Posture Confusion Matrix")
plt.tight_layout()
plt.show()

# Classification report
print("\\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
