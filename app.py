from flask import Flask , jsonify , request , render_template
from PIL import Image
import io
import config
from src.inference import load_model, predict_image

app = Flask(__name__)

load_model(config.CHECKPOINT_PATH, config.CLASS_NAMES)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    file = request.files['image']
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    result = predict_image(image)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)