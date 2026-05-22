# evaluate.py - Structural Metric Tracker Client 27
import tensorflow as tf
import numpy as np
import os
from data_loader import load_data

CLASS_NAMES = ['Moderate', 'Short', 'Tall']
MODEL_PATH  = 'best_model.keras'

def evaluate():
    print("=" * 45)
    print("  📈 EVALUATING WORKSPACE MODULE FOR CLIENT 27  ")
    print("=" * 45)

    if not os.path.exists(MODEL_PATH):
        print("Error: Target checkpoint best_model.keras missing! Run train.py first.")
        return

    model = tf.keras.models.load_model(MODEL_PATH)
    _, val_ds = load_data()

    all_preds = []
    all_labels = []

    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        all_preds.extend(np.argmax(preds, axis=1))
        all_labels.extend(np.argmax(labels.numpy(), axis=1))

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = np.mean(all_preds == all_labels)
    print(f"\nValidation Target Matrix Accuracy Score: {accuracy * 100:.2f}%")

if __name__ == '__main__':
    evaluate()
