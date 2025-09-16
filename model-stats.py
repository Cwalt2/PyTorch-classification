import torch
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from torchinfo import summary
from sklearn.metrics import classification_report, confusion_matrix
import os

# ====================================================================
# CONFIGURATION - CHANGE THESE VALUES
# ====================================================================
# 1. Path to your saved model file
MODEL_PATH = "models/cats_vs_dogs_resnet18_50_epochs.pth"

# 2. Path to your validation data directory (e.g., "data/val")
VAL_DATA_DIR = "data/val"

# 3. Model and data parameters
BATCH_SIZE = 32
NUM_CLASSES = 2 # Number of classes (2 for cats vs. dogs)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ====================================================================


def evaluate_model(model_path, val_dir, num_classes, batch_size, device):
    """
    Loads a trained model and evaluates its performance on a validation set.
    """
    print(f"Using device: {device}")

    # --- 1. Load the Data ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if not os.path.exists(val_dir):
        print(f"Error: Validation directory not found at '{val_dir}'")
        return

    val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    class_names = val_dataset.classes
    print(f"Found classes: {class_names}")


    # --- 2. Load the Model Architecture and Weights ---
    # First, create the model architecture
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, num_classes)

    # Second, load your saved weights into the architecture
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path}'")
        return
        
    model = model.to(device)
    model.eval() # Set the model to evaluation mode


    # --- 3. Get Model Architecture Summary ---
    print("\n==================== MODEL SUMMARY ====================")
    summary(model, input_size=(batch_size, 3, 224, 224), device=str(device))


    # --- 4. Get Performance Metrics ---
    print("\n================== PERFORMANCE METRICS ==================")
    all_preds = []
    all_labels = []

    with torch.no_grad(): # Disable gradient calculation for efficiency
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Generate and print the classification report
    print("\n--- Classification Report ---")
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(report)

    # Generate and print the confusion matrix
    print("\n--- Confusion Matrix ---")
    conf_matrix = confusion_matrix(all_labels, all_preds)
    print(f"      Predicted {class_names[0]} | Predicted {class_names[1]}")
    print("-" * 50)
    print(f"True {class_names[0]:<5} | {conf_matrix[0][0]:^12} | {conf_matrix[0][1]:^12}")
    print(f"True {class_names[1]:<5} | {conf_matrix[1][0]:^12} | {conf_matrix[1][1]:^12}\n")


# --- Run the evaluation ---
if __name__ == '__main__':
    evaluate_model(
        model_path=MODEL_PATH,
        val_dir=VAL_DATA_DIR,
        num_classes=NUM_CLASSES,
        batch_size=BATCH_SIZE,
        device=DEVICE
    )