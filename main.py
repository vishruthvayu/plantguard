from src.dataset import PlantDiseaseDataset
from torch.utils.data import DataLoader
import config

dataset = PlantDiseaseDataset(root_dir=config.DATA_DIR, transform=None)

print(f"Total images: {len(dataset)}")
print(f"Total classes: {len(dataset.classes)}")
print(f"first 3 classes: {dataset.classes[:3]}")
print(f"Sample entry: {dataset.samples[0]}")
