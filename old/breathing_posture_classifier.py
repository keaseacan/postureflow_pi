import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.initializers import HeNormal

# Load data
df = pd.read_excel("breathing_features.xlsx")

# Combine IMF features + one-hot encoded Person
X_imf = df[[f"IMF_{i}" for i in range(1, 10)]].to_numpy()
X_person = pd.get_dummies(df["Person"], prefix="Person").to_numpy()
X = np.concatenate([X_imf, X_person], axis=1)

# Encode posture labels
le = LabelEncoder()
y = le.fit_transform(df["Position"])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Define model
model = Sequential([
    Input(shape=(X.shape[1],)),
    Dense(128, kernel_initializer=HeNormal()),
    LeakyReLU(alpha=0.1),
    Dropout(0.2),
    
    Dense(64, kernel_initializer=HeNormal()),
    LeakyReLU(alpha=0.1),
    Dropout(0.2),
    
    Dense(32, kernel_initializer=HeNormal()),
    LeakyReLU(alpha=0.1),
    Dropout(0.1),

    Dense(len(le.classes_), activation='softmax')
])
# Compile
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Learning rate scheduler
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

# Train
model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=16, callbacks=[lr_scheduler], verbose=1)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest accuracy (with person input): {test_acc:.2f}")

# Confusion matrix
y_pred = model.predict(X_test).argmax(axis=1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Detailed performance
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
