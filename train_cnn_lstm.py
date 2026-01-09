import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense
from tensorflow.keras.optimizers import Adam


# ===============================
# 1. LOAD DATASET
# ===============================
X = np.load("X_ecg.npy")
y = np.load("y_ecg.npy")

print("Dataset loaded:")
print("X shape:", X.shape)
print("y shape:", y.shape)

# ===============================
# 2. TRAIN-TEST SPLIT (STRATIFIED)
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# 3. CNN + LSTM MODEL
# ===============================
model = Sequential([
    Conv1D(32, 3, activation='relu', input_shape=(180,1)),
    MaxPooling1D(2),
    LSTM(64),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ===============================
# 4. HANDLE CLASS IMBALANCE 🔥
# ===============================
class_weight = {
    0: 1.0,
    1: 5.0
}

# ===============================
# 5. TRAIN MODEL
# ===============================
model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    class_weight=class_weight
)

# ===============================
# 6. EVALUATE MODEL
# ===============================
y_prob = model.predict(X_test)
y_pred = (y_prob > 0.3).astype(int)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Normal", "Abnormal"],
    zero_division=0
))


model.save("ecg_cnn_lstm_model.h5")
print("Model saved successfully!")
