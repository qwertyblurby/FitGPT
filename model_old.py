import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import json
from PIL import Image
import os

# Define the neural network architecture
class MyModel(nn.Module):
    def __init__(self, num_classes_per_item):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 47 * 147, 128)  # Adjusted input size after pooling
        self.fc2 = nn.Linear(128, num_classes_per_item[0])  # Output layer for the first item
        self.fc3 = nn.Linear(128, num_classes_per_item[1])  # Output layer for the second item
        self.fc4 = nn.Linear(128, num_classes_per_item[2])  # Output layer for the third item
        self.fc5 = nn.Linear(128, num_classes_per_item[3])  # Output layer for the fourth item

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 32 * 47 * 147)  # Flatten before fully connected layer
        x = nn.functional.relu(self.fc1(x))
        output1 = self.fc2(x)
        output2 = self.fc3(x)
        output3 = self.fc4(x)
        output4 = self.fc5(x)
        return torch.softmax(output1, dim=1), torch.softmax(output2, dim=1), torch.softmax(output3, dim=1), torch.softmax(output4, dim=1)



# Custom dataset class
class MyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = self._load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, label_path = self.data[idx]
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        with open(label_path) as f:
            label_dicts = json.load(f)
            print("Label dictionaries:", label_dicts)
        # Extract labels for each article of clothing
        labels_list = []
        for item in label_dicts:
            labels_list.append(torch.tensor(label_dicts[item], dtype=torch.float32))
        print("Labels list:", labels_list)
        return image, labels_list

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

# Initialize the model
num_classes_per_item = [17, 17, 17, 17]  # Assuming 17 classes for each article of clothing
model = MyModel(num_classes_per_item)

# Initialize the optimizer and criterion
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels_list in train_loader:
        optimizer.zero_grad()
        outputs_list = model(inputs)
        loss = 0
        for outputs, labels in zip(outputs_list, labels_list):
            loss += criterion(outputs, labels.argmax(dim=1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    # Print average training loss per epoch
    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# Run model on sample image
sample_image = Image.open("preprocessed_image.png")
sample_image = transform(sample_image).unsqueeze(0)  # Add batch dimension
model.eval()
with torch.no_grad():
    outputs_list = model(sample_image)
    predicted_probabilities_list = [outputs.squeeze().tolist() for outputs in outputs_list]
    # print("Predicted probabilities for each article of clothing:", predicted_probabilities_list)
    

