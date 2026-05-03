from flask import Flask , jsonify , request
from PIL import Image
import io
import config
from src.dataset import PlantDiseaseDataset
from src.inference import load_model, predict_image

app = Flask(__name__)

dataset = PlantDiseaseDataset(root_dir=config.DATA_DIR, transform=None)
load_model(config.CHECKPOINT_PATH, dataset.classes)

@app.route('/')
def home():
    return "Welcome to the Plant Disease Detection API!"

@app.route('/predict',methods=['POST'])
def predict():
    file = request.files['image']
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    result = predict_image(image)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)