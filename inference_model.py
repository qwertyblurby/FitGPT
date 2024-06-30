import torch
import torch.nn as nn
import torch.nn.functional as F

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
		self.fc_shirt = nn.Linear(128, num_outputs)
		self.fc_outerwear = nn.Linear(128, num_outputs)
		self.fc_pants = nn.Linear(128, num_outputs)
		self.fc_shoes = nn.Linear(128, num_outputs)

	def forward(self, x):
		x = x.reshape(1, 1, 200, 600)
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
		
		shirt_output = F.softmax(self.fc_shirt(x), dim=1)
		outerwear_output = F.softmax(self.fc_outerwear(x), dim=1)
		pants_output = F.softmax(self.fc_pants(x), dim=1)
		shoes_output = F.softmax(self.fc_shoes(x), dim=1)
		return shirt_output, outerwear_output, pants_output, shoes_output