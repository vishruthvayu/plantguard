import torch 

DEVICE = "cuda" if torch.is_vulkan_available() else "cpu"

DATA_DIR = "data/plantvillage dataset/color"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2
LEARNING_RATE = 1e-3

NUM_CLASSES = 38