# train.py - Client 27 Training Loop
import tensorflow as tf
from model import build_model
from data_loader import load_data

EPOCHS = 3

def run_training_pipeline():
    train_ds, val_ds = load_data()
    model, _ = build_model()
    
    print("Training model...")
    model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, verbose=1)
    model.save('best_model.keras')
    print("Saved best_model.keras successfully!")

if __name__ == '__main__':
    run_training_pipeline()
