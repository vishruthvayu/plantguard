import torch
import config
from src.dataset import PlantDiseaseDataset
from torch.utils.data import DataLoader,random_split
from src.model import get_model
from src.train import train_one_epoch,validate
from src.utils import save_checkpoint,load_checkpoint

device = config.DEVICE

dataset = PlantDiseaseDataset(root_dir=config.DATA_DIR, transform=config.TRAIN_TRANSFORMS)

# print(f"Total images: {len(dataset)}")
# print(f"Total classes: {len(dataset.classes)}")
# print(f"first 3 classes: {dataset.classes[:3]}")
# print(f"Sample entry: {dataset.samples[0]}")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset,[train_size,val_size])

train_loader = DataLoader(train_dataset,batch_size=config.BATCH_SIZE,shuffle=True)
val_loader = DataLoader(val_dataset,batch_size=config.BATCH_SIZE,shuffle=False)

# print(f"Train samples: {len(train_dataset)}")
# print(f"validation samples: {len(val_dataset)}")
# images,labels = next(iter(train_loader))
# print(f"Batch image shape: {images.shape}")
# print(f"Batch label shape: {labels.shape}")

model = get_model(num_classes=config.NUM_CLASSES).to(device)

# dummy_input = torch.randn(32,3,224,224)
# output = model(dummy_input)
# print(f"Model output shape: {output.shape}")
# total = sum(p.numel() for p in model.parameters())
# trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"total parameters: {total}")
# print(f"trainable parameters: {trainable}")

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=config.LEARNING_RATE)
best_val_acc = 0.0

for epoch in range(config.NUM_EPOCHS):
    train_loss,train_acc = train_one_epoch(model,train_loader,optimizer,criterion,device)
    val_loss,val_acc = validate(model,val_loader,criterion,device)

    print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
    print(f"Train Loss: {train_loss:.4f} | Train ACC: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f} | Val ACC: {val_acc:.2f}%")
    print("-" * 30)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_checkpoint(model,optimizer,epoch+1,val_acc,"checkpoints/best_model.pth")
        