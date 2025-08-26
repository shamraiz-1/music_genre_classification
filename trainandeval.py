# trainandeval.py
import os
import numpy as np
import librosa
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_PATH = r"C:\Users\ST\Desktop\level 03\genres_original"

def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, duration=30)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

def extract_spectrogram(file_path, img_size=(128, 128)):
    y, sr = librosa.load(file_path, duration=30)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    S_resized = tf.image.resize(S_dB[..., np.newaxis], img_size).numpy()
    return S_resized

def load_dataset(dataset_path, use_cnn=False):
    X, y = [], []
    genres = os.listdir(dataset_path)

    for genre in genres:
        genre_dir = os.path.join(dataset_path, genre)
        for file in os.listdir(genre_dir):
            if file.endswith(".wav"):
                file_path = os.path.join(genre_dir, file)
                try:
                    if use_cnn:
                        features = extract_spectrogram(file_path)
                    else:
                        features = extract_mfcc(file_path)
                    X.append(features)
                    y.append(genre)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    return np.array(X), np.array(y)

def train_rf_xgb(dataset_path):
    print("📥 Loading dataset (MFCC features)...")
    X, y = load_dataset(dataset_path, use_cnn=False)
    le = LabelEncoder()
    y = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print("RF Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

    xgb = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    print("XGB Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

    joblib.dump(rf, "rf_model.pkl")
    joblib.dump(xgb, "xgb_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(le, "label_encoder.pkl")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
    plt.title("Confusion Matrix (XGB)")
    plt.show()

def train_cnn(dataset_path):
    print("📥 Loading dataset (Spectrograms)...")
    X, y = load_dataset(dataset_path, use_cnn=True)
    le = LabelEncoder()
    y = le.fit_transform(y)

    X = X / 255.0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(128, 128, 1),
        include_top=False,
        weights=None
    )

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(len(np.unique(y)), activation="softmax")
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=16)

    model.save("cnn_model.h5")
    joblib.dump(le, "cnn_label_encoder.pkl")

    plt.figure()
    plt.plot(history.history["accuracy"], label="train acc")
    plt.plot(history.history["val_accuracy"], label="val acc")
    plt.legend()
    plt.title("CNN Accuracy")
    plt.show()

if __name__ == "__main__":
    train_rf_xgb(DATASET_PATH)
    train_cnn(DATASET_PATH)
