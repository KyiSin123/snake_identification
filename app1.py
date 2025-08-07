from flask import Flask, render_template, request, redirect
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model once
model = tf.keras.models.load_model('snake_model.h5')  # Your trained model path
class_names = ['Python', 'Cobra', 'Viper', 'Krait', 'Russell\'s Viper']  # Example classes

def predict_snake(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Adjust to model input
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)[0]
    max_index = np.argmax(predictions)
    return class_names[max_index], float(predictions[max_index]) * 100

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            predicted_class, confidence = predict_snake(file_path)
            return render_template('index.html',
                                   image_url=file_path,
                                   prediction=predicted_class,
                                   confidence=confidence)
    return render_template('index.html')