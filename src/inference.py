import torch
from PIL import Image
import config
from src.model import get_model

def predict(image_path,checkpoint_path,class_names):
    model = get_model(num_classes=config.NUM_CLASSES)
    checkpoint = torch.load(checkpoint_path,map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.to(config.DEVICE)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    tensor = config.TEST_TRANSFORMS(image).unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max(1)

    return{
        "class": class_names[predicted.item()],
        "confidence": confidence.item()*100
    }