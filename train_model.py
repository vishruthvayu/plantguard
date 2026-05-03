# train_model.py → run this to train
# main.py → run this for inference
# we run all thr training part in colab using t4 in plantguard notebook.

import torch
import config
from src.dataset import PlantDiseaseDataset
from torch.utils.data import DataLoader, random_split
from src.model import get_model, unfreeze_last_blocks
from src.train import train_one_epoch, validate
from src.utils import save_checkpoint

device = config.DEVICE

# Dataset
dataset = PlantDiseaseDataset(root_dir=config.DATA_DIR, transform=config.TRAIN_TRANSFORMS)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

# Model
model = get_model(num_classes=config.NUM_CLASSES).to(device)
criterion = torch.nn.CrossEntropyLoss()

# optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

# # --- Phase 1: Train classifier only ---
# print("--- Phase 1: Training Classifier ---")
# best_val_acc = 0.0
# for epoch in range(config.NUM_EPOCHS):
#     train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
#     val_loss, val_acc = validate(model, val_loader, criterion, device)

#     print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
#     print(f"Train Loss: {train_loss:.4f} | Train ACC: {train_acc:.2f}%")
#     print(f"Val Loss: {val_loss:.4f} | Val ACC: {val_acc:.2f}%")
#     print("-" * 30)

#     if val_acc > best_val_acc:
#         best_val_acc = val_acc
#         save_checkpoint(model, optimizer, epoch+1, val_acc, "checkpoints/best_model.pth")

# # --- Load existing checkpoint ---
print("--- Loading existing checkpoint ---")
from src.utils import load_checkpoint
checkpoint_path = "checkpoints/best_model.pth"

optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
load_checkpoint(model, optimizer, checkpoint_path)

# --- Phase 2: Fine-tuning ---
print("\n--- Phase 2: Fine-tuning ---")
model = unfreeze_last_blocks(model, num_blocks=4)
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-5
)

best_val_acc = 0.0
for epoch in range(5):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    print(f"Finetune Epoch {epoch+1}/5")
    print(f"Train Loss: {train_loss:.4f} | Train ACC: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f} | Val ACC: {val_acc:.2f}%")
    print("-" * 30)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_checkpoint(model, optimizer, epoch+1, val_acc, "checkpoints/best_model_finetuned.pth")