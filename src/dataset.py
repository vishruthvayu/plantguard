import os
from torch.utils.data import Dataset
from PIL import Image

class PlantDiseaseDataset(Dataset):
    def __init__(self,root_dir,transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir,d))
        ])

        self.class_to_idx = {cls_name: idx for idx,cls_name in enumerate(self.classes)}

        self.samples = []
        for cls_name in self.classes:
            cls_folder = os.path.join(self.root_dir,cls_name)
            label = self.class_to_idx[cls_name]

            for img_file in os.listdir(cls_folder):
                if img_file.endswith((".jpg",".JPG",".jpeg",".JPEG",".png",".PNG")):
                    img_path = os.path.join(cls_folder,img_file)
                    self.samples.append((img_path,label))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self,index):
        img_path, label = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image,label