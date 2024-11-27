import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
IMG_SIZE = 224
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'supersecretkey')  # Default for testing
LOGIN_USERNAME = os.getenv('LOGIN_USERNAME', 'admin')
LOGIN_PASSWORD = os.getenv('LOGIN_PASSWORD', 'password')

# Initialize app and Flask-Login
app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User management (mocked for simplicity)
class User(UserMixin):
    def __init__(self, id):
        self.id = id

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

# Lazy model loading
model = None

def load_model():
    global model
    if model is None:
        model = tf.keras.models.load_model('qr_code_classifier_transfer_learning.h5')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Invalid or corrupted image file.")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

def predict_qr_code(img_path):
    load_model()
    img = preprocess_image(img_path)
    prediction = model.predict(img)
    result = "Real" if prediction[0][0] >= 0.5 else "Fake"
    confidence = prediction[0][0]
    return f"{result} QR Code (Confidence: {confidence:.2f})"

# Routes
@app.route('/')
@login_required
def index():
    return render_template('index.html', username=current_user.id)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Check username and password against environment variables
        if username == LOGIN_USERNAME and password == LOGIN_PASSWORD:
            user = User(id=username)
            login_user(user)
            return redirect(url_for('index'))
        flash('Invalid credentials. Please try again.')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.')
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'files' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    files = request.files.getlist('files')
    results = []
    for file in files:
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        try:
            filename = secure_filename(file.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(img_path)
            result = predict_qr_code(img_path)
            results.append({'filename': filename, 'result': result})
        except Exception as e:
            results.append({'filename': file.filename, 'error': str(e)})
    return render_template('index.html', result=results, username=current_user.id)

@app.route('/scan_qr', methods=['POST'])
@login_required
def scan_qr():
    try:
        # Get the QR code data from the request
        qr_code_data = request.json.get('qr_code')

        if not qr_code_data:
            return jsonify({'error': 'No QR code data received'}), 400

        # Process the QR code data with the model
        result = predict_qr_code_from_string(qr_code_data)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def predict_qr_code_from_string(qr_code_data):
    load_model()
    # Since QR code data is text, we can return the same result without processing as image
    prediction = model.predict(np.array([qr_code_data]))  # Modify as per model requirements
    result = "Real" if prediction[0][0] >= 0.5 else "Fake"
    return f"{result} QR Code (Confidence: {prediction[0][0]:.2f})"

if __name__ == '__main__':
    app.run(debug=True)
