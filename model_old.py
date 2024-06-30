import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import json

# Define the order of colors
color_order = ["black", "gray", "white", "dark_blue", "light_blue", "cyan", "cream", "yellow", "purple", "green", "light_green", "dark_brown", "light_brown", "maroon", "red", "pink"]

# Define the neural network architecture
class MyModel(nn.Module):
    def __init__(self, num_outputs):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=5) # 40x120
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 20x60
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 10x30
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # 5x15
        
        self.conv5 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        self.bn5 = nn.BatchNorm2d(128)
        self.pool5 = nn.AvgPool2d(kernel_size=5, stride=5)
        
        self.fc1 = nn.Linear(128 * 1 * 3, 128)
        self.fc_bn = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.3)
        self.fc_shirt = nn.Linear(128, num_outputs)
        self.fc_outerwear = nn.Linear(128, num_outputs)
        self.fc_pants = nn.Linear(128, num_outputs)
        self.fc_shoes = nn.Linear(128, num_outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool5(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_bn(self.fc1(x)))
        # x = self.dropout(x)
        
        shirt_output = F.softmax(self.fc_shirt(x), dim=1)
        outerwear_output = F.softmax(self.fc_outerwear(x), dim=1)
        pants_output = F.softmax(self.fc_pants(x), dim=1)
        shoes_output = F.softmax(self.fc_shoes(x), dim=1)
        return shirt_output, outerwear_output, pants_output, shoes_output

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
            labels = json.load(f)
        return image, labels

    def _load_data(self):
        data = []
        for filename in os.listdir(self.root_dir):
            if filename.endswith(".png"):
                img_path = os.path.join(self.root_dir, filename)
                label_path = os.path.join(self.root_dir, f"{os.path.splitext(filename)[0]}.json")
                if os.path.exists(label_path):
                    data.append((img_path, label_path))
        return data

# Function to convert labels to tensors
def labels_to_tensor(labels_dict):
    tensor_list = []
    for color, tensor in labels_dict.items():
        tensor_list.append(tensor)
    stacked_tensor = torch.stack(tensor_list, dim=1)
    return stacked_tensor

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        # Convert data
        labels = {article: {color: tensor.float() for color, tensor in CTdict.items()} for article, CTdict in labels.items()}
        shirt_target, outerwear_target, pants_target, shoes_target = map(
            lambda x: labels_to_tensor(x).to(device),
            (labels['shirt'], labels['outerwear'], labels['pants'], labels['shoes'])
        )
        inputs = inputs.to(device)
        optimizer.zero_grad()
        shirt_output, outerwear_output, pants_output, shoes_output = model(inputs)
        shirt_loss = criterion(shirt_output, shirt_target)
        outerwear_loss = criterion(outerwear_output, outerwear_target)
        pants_loss = criterion(pants_output, pants_target)
        shoes_loss = criterion(shoes_output, shoes_target)
        loss = shirt_loss + outerwear_loss + pants_loss + shoes_loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        
        # Calculate accuracy
        _, preds_shirt = torch.max(shirt_output, 1)
        _, preds_outerwear = torch.max(outerwear_output, 1)
        _, preds_pants = torch.max(pants_output, 1)
        _, preds_shoes = torch.max(shoes_output, 1)
        _, truth_shirt = torch.max(shirt_target, 1)
        _, truth_outerwear = torch.max(outerwear_target, 1)
        _, truth_pants = torch.max(pants_target, 1)
        _, truth_shoes = torch.max(shoes_target, 1)
        correct += (preds_shirt == truth_shirt).sum().item()
        correct += (preds_outerwear == truth_outerwear).sum().item()
        correct += (preds_pants == truth_pants).sum().item()
        correct += (preds_shoes == truth_shoes).sum().item()
        total += truth_shirt.size(0) + truth_outerwear.size(0) + truth_pants.size(0) + truth_shoes.size(0)
            
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch}, Loss: {epoch_loss*100:.2f} Acc: {correct/total*100:.2f}")

def test(model, device, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Convert data
            labels = {article: {color: tensor.float() for color, tensor in CTdict.items()} for article, CTdict in labels.items()}
            shirt_target, outerwear_target, pants_target, shoes_target = map(
                lambda x: labels_to_tensor(x).to(device),
                (labels['shirt'], labels['outerwear'], labels['pants'], labels['shoes'])
            )
            inputs = inputs.to(device)
            
            shirt_output, outerwear_output, pants_output, shoes_output = model(inputs)
            shirt_loss = criterion(shirt_output, shirt_target)
            outerwear_loss = criterion(outerwear_output, outerwear_target)
            pants_loss = criterion(pants_output, pants_target)
            shoes_loss = criterion(shoes_output, shoes_target)
            loss = shirt_loss + outerwear_loss + pants_loss + running_loss
            running_loss += loss.item() * inputs.size(0)
            
            # Calculate accuracy
            _, preds_shirt = torch.max(shirt_output, 1)
            _, preds_outerwear = torch.max(outerwear_output, 1)
            _, preds_pants = torch.max(pants_output, 1)
            _, preds_shoes = torch.max(shoes_output, 1)
            _, truth_shirt = torch.max(shirt_target, 1)
            _, truth_outerwear = torch.max(outerwear_target, 1)
            _, truth_pants = torch.max(pants_target, 1)
            _, truth_shoes = torch.max(shoes_target, 1)
            correct += (preds_shirt == truth_shirt).sum().item()
            correct += (preds_outerwear == truth_outerwear).sum().item()
            correct += (preds_pants == truth_pants).sum().item()
            correct += (preds_shoes == truth_shoes).sum().item()
            total += truth_shirt.size(0) + truth_outerwear.size(0) + truth_pants.size(0) + truth_shoes.size(0)
            
        epoch_loss = running_loss / len(test_loader.dataset)
        print(f"Validation set loss: {epoch_loss*100:.2f} Acc: {correct/total*100:.2f}")

def main():
    # Define transformations
    train_transform = transforms.Compose([
        transforms.Resize((200, 600)),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])
    transform = transforms.Compose([
        transforms.Resize((200, 600)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    train_dataset = MyDataset(root_dir='training_set', transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_dataset = MyDataset(root_dir='validation_set', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=13, shuffle=False)

    # Initialize the model
    model = MyModel(len(color_order)).to(device)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    test_criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    # optimizer = optim.Adagrad(model.parameters(), lr=0.001)
    # optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7)
    # Softmax losses. Adam: 28.0 / 10.0    Adagrad: 28.2 / 9.7    RMSprop: 29.1 / 10.0    SGD: 29.1 / 10.0
    # ReLU losses.    Adam: 11.7 / 8.7     Adagrad: 18.8 / 8.7    RMSprop: 16.3 / 8.9     SGD: 13.6 / 10.3
    
    # Training loop
    num_epochs = 25
    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        test(model, device, test_loader, test_criterion)
        scheduler.step()
    
    if input("Save model? (y/n) ").strip().lower() == "y":
        torch.save(model.state_dict(), "fitgpt_model.pt")

if __name__ == "__main__":
    main()
