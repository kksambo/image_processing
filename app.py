from flask import *
from flask_cors import CORS

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Initialize the Flask application
app = Flask(__name__)
CORS(app)  # Enable CORS for the entire app

# Load the pre-trained MobileNetV2 model (pre-trained on ImageNet)
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

# Filter the predictions to look for "can" or "plastic"
def filter_for_can_or_plastic(decoded_prediction):
    possible_labels = ['can', 'cup', 'bottle', 'container', 'plastic', 'jar', 'vending_machine']
    for (_, label, confidence) in decoded_prediction:
        if any(term in label.lower() for term in possible_labels):
            return {"label": label, "confidence": float(confidence)}
    return {"label": "Unknown", "confidence": 0.0}

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        if not os.path.exists('uploads'):
            os.makedirs('uploads')

        img_path = os.path.join('uploads', file.filename)
        file.save(img_path)

        img_array = preprocess_image(img_path)
        prediction = model.predict(img_array)
        decoded_prediction = tf.keras.applications.mobilenet_v2.decode_predictions(prediction, top=3)[0]
        print("Decoded Predictions:", decoded_prediction)
        result = filter_for_can_or_plastic(decoded_prediction)
        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)