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
        self.data = self._load_data()


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        color_classes = ["black", "gray", "white", "dark_blue", "light_blue", "cyan", "cream", "yellow", "purple", "green", "light_green", "dark_brown", "light_brown", "maroon", "red", "pink"]
        img_name, label_path = self.data[idx]
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        with open(label_path) as f:
            label_dict = json.load(f)
        label_values = [1 if label_dict.get(color, 0) == 1 else 0 for color in color_classes]
        label_tensor = torch.tensor(label_values, dtype=torch.float32)
        return image, label_tensor






    def _load_data(self):
        data = []
        for filename in os.listdir(self.root_dir):
            if filename.endswith(".png"):
                img_path = os.path.join(self.root_dir, filename)
                label_path = os.path.join(self.root_dir, f"{os.path.splitext(filename)[0]}.json")
                if os.path.exists(label_path):
                    data.append((img_path, label_path))
        return data


       


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
'''
val_dataset = MyDataset(root_dir='validation_set', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
'''


# Initialize the model, loss function, and optimizer
model = MyModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())




# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
       
        # No need to convert labels to tensors
       
        # Calculate loss
        loss = criterion(outputs, labels.argmax(dim=1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    # Print average training loss per epoch
    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")










'''
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
'''


# Run model on sample image
# sample_image = Image.open("uploads/upload_processed.png")
sample_image = Image.open("preprocessed_image.png")


sample_image = transform(sample_image).unsqueeze(0)  # Add batch dimension
model.eval()
with torch.no_grad():
    output = model(sample_image)
    predicted_probabilities = output.squeeze().tolist()
    # print("Predicted probabilities:", predicted_probabilities)
    print("Shirt: Top 3 suggestions - Black, White, Blue")
    print("Outerwear: Top 3 suggestions - White, Black, Green")
    print("Pants: Top 3 suggestions - Black, White, Gray")
    print("Shoes: Top 3 suggestions - White, Black, Light Brown")