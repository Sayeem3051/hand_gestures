import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical

# Path to dataset (update if your folder name is slightly different)
DATASET_PATH = r"C:\Users\shaba\program\handgesture\Sign-Language-Digits-Dataset"

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

# Function to extract landmarks from an image
def extract_landmarks(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"[Warning] Could not read image: {image_path}")
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
    return None

# Function to load dataset
def load_dataset(split="train"):
    X, y = [], []
    split_path = os.path.join(DATASET_PATH, split)
    print(f"[Info] Loading data from: {split_path}")
    for label_folder in sorted(os.listdir(split_path)):
        label_path = os.path.join(split_path, label_folder)
        if not os.path.isdir(label_path):
            continue
        try:
            label = int(label_folder)  # Folder names: '0', '1', ..., '9'
        except ValueError:
            print(f"[Warning] Skipping non-numeric folder: {label_folder}")
            continue
        for img_file in os.listdir(label_path):
            img_path = os.path.join(label_path, img_file)
            landmarks = extract_landmarks(img_path)
            if landmarks is not None:
                X.append(landmarks)
                y.append(label)
    return np.array(X), np.array(y)

# Load train, validation, and test sets
X_train, y_train = load_dataset("train")
X_val, y_val = load_dataset("valid")
X_test, y_test = load_dataset("test")

# Reshape and normalize
X_train = X_train.reshape(-1, 63, 1)
X_val = X_val.reshape(-1, 63, 1)
X_test = X_test.reshape(-1, 63, 1)

# One-hot encode labels
y_train = to_categorical(y_train, num_classes=10)
y_val = to_categorical(y_val, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Build 7-layer CNN model
model = Sequential([
    Conv1D(32, 3, activation='relu', input_shape=(63, 1)),
    MaxPooling1D(2),
    Conv1D(64, 3, activation='relu'),
    MaxPooling1D(2),
    Conv1D(128, 3, activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

print("[Info] Training model...")
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"[Result] Test accuracy: {test_acc:.2f}")

# Save model
model.save("hand_sign_cnn_model.h5")
print("[Info] Model saved as 'hand_sign_cnn_model.h5'")
