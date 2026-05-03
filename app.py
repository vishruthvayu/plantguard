from flask import Flask, jsonify, request, render_template
from PIL import Image
import io
import os
import torch
import gdown
import config
from src.inference import load_model, predict_image

torch.set_num_threads(1) 
app = Flask(__name__)

def download_model():
    os.makedirs("checkpoints", exist_ok=True)
    if not os.path.exists(config.CHECKPOINT_PATH):
        print("Downloading model from Google Drive...")
        url = "https://drive.google.com/uc?export=download&id=1QfS1RoB3T6N1Cb7I0x4FjVbzKbcaCtL0"
        gdown.download(url, config.CHECKPOINT_PATH, quiet=False)
        print("Model downloaded successfully!")

download_model()
load_model(config.CHECKPOINT_PATH, config.CLASS_NAMES)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    result = predict_image(image)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)