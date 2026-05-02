import torch
import os
def save_checkpoint(model,optimizer,epoch,val_acc,filename):
    checkpoint = {
        "epoch":epoch,
        "model_state":model.state_dict(),
        "optimizer_state":optimizer.state_dict(),
        "val_acc":val_acc
    }
    torch.save(checkpoint,filename)
    print(f"checkpoint saved: {filename}")

def load_checkpoint(model,optimizer,filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    print(f"resume from epoch {checkpoint['epoch']} with val_acc {checkpoint['val_acc']:.2f}%")
    return checkpoint["epoch"],checkpoint["val_acc"]
