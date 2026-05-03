# 🌿 PlantGuard — Plant Disease Detection AI

PlantGuard is a deep learning-powered web application that detects plant diseases from leaf images with **99.47% accuracy** across 38 disease classes.

🔗 **Live Demo:** https://plantguard-2met.onrender.com

> ⚠️ First load may take 50-60 seconds as the free tier wakes up from sleep.

---

## 📸 Demo

Upload a leaf image → Get instant disease detection with confidence score.

---

## ✨ Features

- Detects 38 plant diseases across 14 plant species
- 99.47% validation accuracy
- Real-time prediction with confidence score
- Clean, responsive web interface
- REST API for programmatic access

---

## 🧠 How it Works

1. User uploads a leaf image
2. Flask API receives the image
3. EfficientNet-B0 model processes it
4. Disease class and confidence score returned
5. Result displayed on screen

**Two-stage training:**

- Stage 1: Transfer learning — only classifier layer trained → 97.42%
- Stage 2: Fine-tuning — last 4 EfficientNet blocks unfrozen → 99.47%

---

## 🛠 Tech Stack

| Layer      | Technology                      |
| ---------- | ------------------------------- |
| Model      | PyTorch, EfficientNet-B0        |
| Training   | Transfer Learning + Fine-tuning |
| Backend    | Flask, Gunicorn                 |
| Frontend   | HTML, CSS, JavaScript           |
| Dataset    | PlantVillage (54,305 images)    |
| Deployment | Render                          |

---

## 📊 Training Results

| Phase             | Epochs | Val Accuracy |
| ----------------- | ------ | ------------ |
| Transfer Learning | 10     | 97.42%       |
| Fine-tuning       | 5      | **99.47%**   |

---

## 🚀 Setup & Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/vishruthvayu/plantguard.git
cd plantguard
```

### 2. Create conda environment

```bash
conda create -n plantguard python=3.11
conda activate plantguard
```

### 3. Install dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 4. Download dataset

Download PlantVillage dataset from Kaggle:

```bash
pip install kagglehub
python3 -c "import kagglehub; kagglehub.dataset_download('abdallahalidev/plantvillage-dataset')"
```

### 5. Run the app

```bash
python3 app.py
```

### 6. Open in browser

```
http://127.0.0.1:5000
```

> The model downloads automatically from Google Drive on first run (~47MB)

---

## 🌿 Supported Plants & Diseases

| Plant      | Diseases                                                                                                                                 |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Apple      | Apple Scab, Black Rot, Cedar Apple Rust, Healthy                                                                                         |
| Tomato     | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |
| Potato     | Early Blight, Late Blight, Healthy                                                                                                       |
| Corn       | Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy                                                                         |
| Grape      | Black Rot, Esca, Leaf Blight, Healthy                                                                                                    |
| Peach      | Bacterial Spot, Healthy                                                                                                                  |
| Cherry     | Powdery Mildew, Healthy                                                                                                                  |
| Pepper     | Bacterial Spot, Healthy                                                                                                                  |
| Strawberry | Leaf Scorch, Healthy                                                                                                                     |
| Orange     | Haunglongbing                                                                                                                            |
| Blueberry  | Healthy                                                                                                                                  |
| Raspberry  | Healthy                                                                                                                                  |
| Soybean    | Healthy                                                                                                                                  |
| Squash     | Powdery Mildew                                                                                                                           |

---

## 📁 Project Structure

```
plantguard/
├── app.py              # Flask API
├── train_model.py      # Training script
├── config.py           # Configuration & hyperparameters
├── requirements.txt    # Dependencies
├── render.yaml         # Render deployment config
├── src/
│   ├── dataset.py      # Custom PyTorch Dataset
│   ├── model.py        # EfficientNet model
│   ├── train.py        # Training & validation loops
│   ├── inference.py    # Prediction functions
│   └── utils.py        # Checkpoint save/load
├── templates/
│   └── index.html      # Frontend
└── static/
    └── css/
        └── style.css   # Styling
```

---

## 👨‍💻 Author

Built by **Vishruth Vayu**

This project covers end-to-end ML engineering:

- Custom PyTorch Dataset
- Transfer Learning & Fine-tuning
- Flask REST API
- Frontend Development
- Cloud Deployment
