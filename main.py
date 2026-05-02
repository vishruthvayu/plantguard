from src.dataset import PlantDiseaseDataset
from torch.utils.data import DataLoader,random_split
import config

dataset = PlantDiseaseDataset(root_dir=config.DATA_DIR, transform=config.TRAIN_TRANSFORMS)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset,[train_size,val_size])

train_loader = DataLoader(train_dataset,batch_size=config.BATCH_SIZE,shuffle=True)
val_loader = DataLoader(val_dataset,batch_size=config.BATCH_SIZE,shuffle=False)

print(f"Train samples: {len(train_dataset)}")
print(f"validation samples: {len(val_dataset)}")

images,labels = next(iter(train_loader))
print(f"Batch image shape: {images.shape}")
print(f"Batch label shape: {labels.shape}")

# print(f"Total images: {len(dataset)}")
# print(f"Total classes: {len(dataset.classes)}")
# print(f"first 3 classes: {dataset.classes[:3]}")
# print(f"Sample entry: {dataset.samples[0]}")
