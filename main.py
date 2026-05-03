import config
from src.dataset import PlantDiseaseDataset
from src.inference import predict

# Get class names from dataset
dataset = PlantDiseaseDataset(root_dir=config.DATA_DIR, transform=None)

# Grab first image path from dataset to test
image_path = dataset.samples[0][0]
true_label = dataset.classes[dataset.samples[0][1]]

# Predict
result = predict(
    image_path=image_path,
    # checkpoint_path="checkpoints/best_model.pth",
    checkpoint_path="checkpoints/best_model_finetuned.pth",
    class_names=dataset.classes
)

print(f"True Label:  {true_label}")
print(f"Predicted:   {result['class']}")
print(f"Confidence:  {result['confidence']:.2f}%")