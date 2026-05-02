import torch 
from torchvision import transforms
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = "data/plantvillage dataset/color"

NUM_EPOCHS = 10
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2
LEARNING_RATE = 1e-3

NUM_CLASSES = 38

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])

TEST_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])

