"""

--- DDD SYSTEM TRAINING SCRIPT: READ ME BEFORE RUNNING ---

1. DATASET DIRECTORY SETUP:
   - Ensure you have a folder named 'DDD' in the same directory as this script.
   - Alternatively, update the 'DATASET_PATH' variable to point to your specific folder.
   - The 'DDD' folder MUST contain exactly two subfolders: 'Drowsy' and 'Non Drowsy'.
   - This script uses 'ImageFolder' logic, which automatically assigns labels (0 for Drowsy, 1 for Non Drowsy) 
     based on these folder names.

2. DATASET SPECIFICATIONS:
   - This script is designed specifically for the Driver Drowsiness Dataset (DDD).
   - Total Size: Approximately 2.57GB of raw image data.
   - Because of the large volume of high-resolution images, processing times for each epoch 
     may be significant.

3. EXPECTED RESULTS:
   - While the initial epochs take time, you should observe an iterative improvement in model 
     accuracy and a decrease in training loss with each subsequent epoch.
   - This script uses Transfer Learning with MobileNetV3 Large to provide a robust, 
     high-accuracy model with a small 5-million-parameter footprint.

DATASET SOURCE: https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
import os
import matplotlib.pyplot as plt
import datetime

# --- 1. SETUP & UI ---
print("--- Binary Drowsiness Trainer (Folder Based) ---")
user_epochs = int(input("Enter number of epochs: "))
run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M")
save_dir = f"Test_Run_Binary_{run_id}_{user_epochs}epochs"
os.makedirs(save_dir, exist_ok=True)

# Update these to your new folder paths
DATASET_PATH = 'DDD' 

# Standard transforms for MobileNet
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 2. DATA LOADING ---

full_dataset = datasets.ImageFolder(root=DATASET_PATH, transform=data_transforms)

# Split into Train (80%) and Test (20%)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_data, test_data = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

print(f"Dataset loaded: {len(full_dataset)} images.")
print(f"Classes found: {full_dataset.classes}")

# --- 3. MODEL SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v3_large(weights='DEFAULT')


model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# --- 4. TRAINING LOOP ---
def run_train():
    train_losses, val_accs = [], []
    
    for epoch in range(user_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
        
        acc = 100 * correct / total
        val_accs.append(acc)
        print(f"Epoch {epoch+1}/{user_epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")

    # Save
    torch.save(model.state_dict(), os.path.join(save_dir, 'binary_model.pth'))
    
    # Plotting
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1); plt.plot(train_losses); plt.title("Training Loss")
    plt.subplot(1, 2, 2); plt.plot(val_accs); plt.title("Test Accuracy")
    plt.savefig(os.path.join(save_dir, "results_graph.png"))
    print(f"Results saved to {save_dir}")

if __name__ == "__main__":
    run_train()
