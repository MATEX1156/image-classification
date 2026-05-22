# data_loader.py - Client 27 Preprocessing Engine
import tensorflow as tf
import os

IMG_SIZE = 224
BATCH_SIZE = 4
CLASS_NAMES = ['moderate', 'short', 'tall']

def load_data(data_dir='dataset'):
    train_dir = os.path.join(data_dir, 'train')
    val_dir   = os.path.join(data_dir, 'val')

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        shuffle=True,
        seed=87,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )

    val_dataset = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        shuffle=False,
        seed=87,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )

    rescale_layer = tf.keras.layers.Rescaling(1.0 / 255)
    train_dataset = train_dataset.map(lambda x, y: (rescale_layer(x), y))
    val_dataset   = val_dataset.map(lambda x, y: (rescale_layer(x), y))
    return train_dataset, val_dataset
