import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os

# ========================
# CONFIG
# ========================
data_dir = "/kagglehub/datasets/biaiscience/dogs-vs-cats/versions/1"  # change to your dataset path
batch_size = 16 # change for higher power cpu/gpu
num_epochs = 2 # change to train for longer
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # cuda = nvidia gpu

# ========================
# DATASET & DATALOADERS
# ========================
transform = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])
}

train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform["train"])
val_dataset   = datasets.ImageFolder(os.path.join(data_dir, "val"),   transform=transform["val"])

# add num_workers if on linux to increase gpu usage
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ========================
# MODEL
# ========================
# Use pretrained ResNet18
model = models.resnet18(pretrained=True) # change to resnet50 for more advanced training

# freeze base layers (optional)
for param in model.parameters():
    param.requires_grad = False  

# replace final layer for 2 classes
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

model = model.to(device)


# ========================
# LOSS & OPTIMIZER
# ========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

# ========================
# TRAINING LOOP
# ========================
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")

    # ---- Training ----
    model.train()
    running_loss, running_corrects = 0.0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)
    print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    # ---- Validation ----
    model.eval()
    val_loss, val_corrects = 0.0, 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)

    val_loss /= len(val_dataset)
    val_acc = val_corrects.double() / len(val_dataset)
    print(f"Val   Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

# ========================
# SAVE MODEL
# ========================
torch.save(model.state_dict(), "models/cats_vs_dogs_resnet18.pth")
print("\nModel saved to cats_vs_dogs_resnet18.pth")
