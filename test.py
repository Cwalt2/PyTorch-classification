import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ========================
# CONFIG
# ========================
model_path = "models/cats_vs_dogs_resnet18_50_epochs.pth" # model path
image_path = "test_images/cutecat.jpeg"  # test image path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================
# PREPROCESSING
# ========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])

# ========================
# LOAD MODEL
# ========================
model = models.resnet18(pretrained=True)   # change to model trained on (ex. resnet50)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)       # 2 classes (cat, dog)

model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# ========================
# PREDICT
# ========================
img = Image.open(image_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(img_tensor)
    _, preds = torch.max(outputs, 1)

classes = ["cat", "dog"]
print(f"Prediction: {classes[preds.item()]}")
