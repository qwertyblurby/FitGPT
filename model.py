import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import json
from PIL import Image
import os

# Define the neural network architecture
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 47 * 147, 128)  # Adjusted input size after pooling
        self.fc2 = nn.Linear(128, 17)  # Output layer with 17 classes (for each color)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 32 * 47 * 147)  # Flatten before fully connected layer
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=1)  # Apply softmax activation

# Custom dataset class
class MyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, f"{idx+1}.png")
        image = Image.open(img_name)
        label_name = os.path.join(self.root_dir, f"{idx+1}.json")
        with open(label_name) as f:
            label = json.load(f)
        if self.transform:
            image = self.transform(image)
        return image, label

# Define transformations
transform = transforms.Compose([
    transforms.Resize((200, 600)),
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.ToTensor(),
])

# Load datasets
train_dataset = MyDataset(root_dir='training_set', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Validation dataset and loader
val_dataset = MyDataset(root_dir='validation_set', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Initialize the model, loss function, and optimizer
model = MyModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        # Calculate loss
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    # Print average training loss per epoch
    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# Validation loop
model.eval()
total_correct = 0
total_samples = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)
val_accuracy = total_correct / total_samples
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Run model on sample image
sample_image = Image.open("uploads/upload_processed.png")
sample_image = transform(sample_image).unsqueeze(0)  # Add batch dimension
model.eval()
with torch.no_grad():
    output = model(sample_image)
    predicted_probabilities = output.squeeze().tolist()
    print("Predicted probabilities:", predicted_probabilities)
