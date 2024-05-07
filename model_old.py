import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import json
from PIL import Image
import os
import preprocessor
from preprocessor import cv2



# Define the order of colors
color_order = ["black", "gray", "white", "dark_blue", "light_blue", "cyan", "cream", "yellow", "purple", "green", "light_green", "dark_brown", "light_brown", "maroon", "red", "pink"]


# Define the neural network architecture
class MyModel(nn.Module):
    def __init__(self, num_outputs):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 47 * 147, 128)  # Adjusted input size after pooling
        self.fc_shirt = nn.Linear(128, num_outputs)
        self.fc_outerwear = nn.Linear(128, num_outputs)
        self.fc_pants = nn.Linear(128, num_outputs)
        self.fc_shoes = nn.Linear(128, num_outputs)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 32 * 47 * 147)  # Flatten before fully connected layer
        x = nn.functional.relu(self.fc1(x))
        shirt_output = nn.functional.softmax(self.fc_shirt(x))
        outerwear_output = nn.functional.softmax(self.fc_outerwear(x))
        pants_output = nn.functional.softmax(self.fc_pants(x))
        shoes_output = nn.functional.softmax(self.fc_shoes(x))
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
        running_loss += loss.item() # * inputs.size(0) DataLoader.batch_size
    # epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch {epoch}, Loss: {running_loss:.4f}")

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0.0
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
            test_loss += criterion(shirt_output, shirt_target).item()
            test_loss += criterion(outerwear_output, outerwear_target).item()
            test_loss += criterion(pants_output, pants_target).item()
            test_loss += criterion(shoes_output, shoes_target).item()
    
    test_loss /= len(test_loader.dataset)
    print(f"Validation set loss: {test_loss:.4f}")

def main():
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((200, 600)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    train_dataset = MyDataset(root_dir='training_set', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataset = MyDataset(root_dir='validation_set', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=13, shuffle=False)

    # Initialize the model
    num_outputs = len(train_dataset[0][1]['shirt'])  # Assuming all clothing items have the same number of color categories
    model = MyModel(num_outputs).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    test_criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.Adam(model.parameters())

    # Training loop
    num_epochs = 30
    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        test(model, device, test_loader, test_criterion)
    
    run_on_image = False
    if run_on_image:
        IMAGENAME = "64.jpg"
        image = cv2.imread(f"uploads/{IMAGENAME}")
        preprocessed_path = f"uploads_processed/{IMAGENAME}"
        preprocessor.preprocess(image, preprocessed_path)
        print("Processed image!")
        
        image = Image.open(preprocessed_path)
        image = transform(image)
        model.eval()
        with torch.no_grad():
            shirt_output, outerwear_output, pants_output, shoes_output = model(image)
        
        for article, article_output in (
            ("Shirt", shirt_output),
            ("Outerwear", outerwear_output),
            ("Pants", pants_output),
            ("Shoes", shoes_output)):
            print(f"{article} probabilities:")
            for color, prob in zip(color_order, article_output[0].tolist()):
                print(f"{color}: {round(100*prob)}%")

if __name__ == "__main__":
    main()
