import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import cv2
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the rice classifier model
try:
    model = tf.keras.models.load_model('rice.h5', custom_objects={'KerasLayer': hub.KerasLayer})
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/details')
def details():
    return render_template('details.html')

@app.route('/result', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('error.html', message="Please upload an image from the home page.")

    if model is None:
        return render_template('error.html', message="Model could not be loaded.")

    if 'image' not in request.files:
        return render_template('error.html', message="No image file provided.")

    f = request.files['image']

    if f.filename == '':
        return render_template('error.html', message="No selected file.")

    try:
        filename = secure_filename(f.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(file_path)

        img = cv2.imread(file_path)
        if img is None:
            return render_template('error.html', message="Invalid image format.")

        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)
        pred_class = np.argmax(pred)
        confidence = round(float(np.max(pred)) * 100, 2)

        class_map = {
            0: 'arborio',
            1: 'basmati',
            2: 'ipsala',
            3: 'jasmine',
            4: 'karacadag'
        }

        predicted_label = class_map.get(pred_class, "Unknown")

        return render_template(
            'results.html',
            prediction_text=predicted_label,
            image_filename=filename,
            confidence_score=confidence
        )

    except Exception as e:
        return render_template('error.html', message=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)