from flask import Flask, request, render_template
import os
import tensorflow as tf
import numpy as np
import cv2
import urllib.request
from model import model1

app = Flask(__name__)
app.secret_key = os.urandom(24)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


modelll=model1

# Fetch and load the labels
KINETICS_URL = "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map.txt"
with urllib.request.urlopen(KINETICS_URL) as f:
    labels = [line.strip().decode('utf-8') for line in f.readlines()]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_video(path, max_frames=64, resize=(224, 224)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]  # BGR to RGB
            frames.append(frame)
            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames) / 255.0

def predict(video_path):
    sample_video = load_video(video_path)
    model_input = tf.constant(sample_video, dtype=tf.float32)[tf.newaxis, ...]

    logits = modelll(model_input)['default'][0]
    probabilities = tf.nn.softmax(logits)
    top_5_indices = np.argsort(probabilities)[::-1][:1]
    top_5_predictions = {labels[i]: probabilities[i].numpy() for i in top_5_indices}
    
    return top_5_predictions

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'video' not in request.files:
            return 'No file part'

        file = request.files['video']

        if file.filename == '':
            return 'No selected file'

        if file and allowed_file(file.filename):
            filename = file.filename
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
            file.save(video_path)

            predictions = predict(video_path)

            return render_template('results.html', predictions=predictions)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)