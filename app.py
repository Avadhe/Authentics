import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify

# Set image size and lazy-load the model
IMG_SIZE = 224
model = None  # Initialize model as None

# Flask app setup
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model lazily to avoid memory spikes
def load_model():
    global model
    if model is None:
        model = tf.keras.models.load_model("qr_code_classifier_transfer_learning.h5")

# Preprocess image
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Image not found or invalid path.")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0  # Normalize
    return np.expand_dims(img, axis=0)  # Add batch dimension

# Predict function
def predict_qr_code(img_path):
    load_model()
    img = preprocess_image(img_path)
    prediction = model.predict(img)
    result = "Real" if prediction[0][0] >= 0.5 else "Fake"
    confidence = prediction[0][0]
    return f"{result} QR Code (Confidence: {confidence:.2f})"

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        img_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(img_path)
        result = predict_qr_code(img_path)
        os.remove(img_path)  # Cleanup uploaded file
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
