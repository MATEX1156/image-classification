# predict.py - Inference Endpoint Client 27
import tensorflow as tf
import numpy as np

CLASS_NAMES = ['Moderate', 'Short', 'Tall']

def predict_single_image(image_path):
    model = tf.keras.models.load_model('best_model.keras')
    img_raw = tf.io.read_file(image_path)
    img = tf.io.decode_image(img_raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [224, 224]) / 255.0
    img_batch = tf.expand_dims(img, axis=0)
    
    predictions = model.predict(img_batch, verbose=0)[0]
    predicted_idx = np.argmax(predictions)
    return CLASS_NAMES[predicted_idx], float(predictions[predicted_idx] * 100)
