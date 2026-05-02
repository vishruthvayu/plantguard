import torch
from tqdm import tqdm

def train_one_epoch(model, loader,optimizer,criterion,device):
    model.train()
    running_loss=0.0
    correct=0
    total=0
    for images,labels in tqdm(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output,labels)
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        _,predicted = output.max(1)
        total+=labels.size(0)
        correct+=predicted.eq(labels).sum().item()

    avg_loss = running_loss/len(loader)
    accuracy = 100.0 * correct/total
    return avg_loss,accuracy

def validate(model,loader,criterion,device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images,labels in tqdm(loader):
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            loss = criterion(output,labels)

            running_loss+=loss.item()
            _,predicted = output.max(1)
            total+=labels.size(0)
            correct+=predicted.eq(labels).sum().item()

    avg_loss = running_loss/len(loader)
    accuracy = 100.0 * correct/total
    return avg_loss,accuracy
