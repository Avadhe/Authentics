import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify

# Set image size
IMG_SIZE = 224

# Load the trained model
model = tf.keras.models.load_model("D:/ML_CASE/qr_code_classifier_transfer_learning.h5")

# Create Flask app
app = Flask(__name__)

# Ensure uploads folder exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to preprocess the image
def preprocess_image(img_path):
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None, "Image not found or invalid path."
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = np.array(img) / 255.0  # Normalize pixel values
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img, None
    except Exception as e:
        return None, str(e)

# Function to predict if the image is real or fake
def predict_qr_code(img_path):
    # Preprocess the image
    img, error = preprocess_image(img_path)
    if error:
        return f"Error during preprocessing: {error}"
    
    # Predict using the model
    try:
        prediction = model.predict(img)
        result = "Real" if prediction[0][0] >= 0.5 else "Fake"
        confidence = prediction[0][0]
        return f"{result} QR Code (Confidence: {confidence:.2f})"
    except Exception as e:
        return f"Error during prediction: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Save the file to the uploads folder
        img_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(img_path)

        # Make the prediction
        result = predict_qr_code(img_path)

        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
