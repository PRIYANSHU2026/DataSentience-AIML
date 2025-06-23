import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet18, ResNet18_Weights

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
train_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomCrop(48, padding=4),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

# Load datasets
train_dataset = datasets.ImageFolder("data/data/train", transform=train_transform)
test_dataset = datasets.ImageFolder("data/data/test", transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# Model definition with pretrained weights and dropout
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # pretrained boost
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)  # handle grayscale

# Add dropout to reduce overfitting
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(512, len(train_dataset.classes))
)

model = model.to(device)

# Loss, optimizer, scheduler
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Training loop
for epoch in range(25):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()  # Step the LR scheduler

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f} | Accuracy = {accuracy:.2f}%")

# Save model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/face_emotion_model.pth")
print("✅ Model saved to models/face_emotion_model.pth")
