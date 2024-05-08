import torch
import torch.nn as nn

class MyModel(nn.Module):
	def __init__(self, num_outputs):
		super(MyModel, self).__init__()
		self.conv1 = nn.Conv2d(1, 2, kernel_size=5)
		self.pool = nn.MaxPool2d(kernel_size=4, stride=4)
		self.fc1 = nn.Linear(2 * 49 * 149, 64)
		self.fc_shirt = nn.Linear(64, num_outputs)
		self.fc_outerwear = nn.Linear(64, num_outputs)
		self.fc_pants = nn.Linear(64, num_outputs)
		self.fc_shoes = nn.Linear(64, num_outputs)

	def forward(self, x):
		x = self.pool(nn.functional.relu(self.conv1(x)))
		x = x.view(-1, 2 * 49 * 149)
		x = nn.functional.relu(self.fc1(x))
		shirt_output = nn.functional.softmax(self.fc_shirt(x))
		outerwear_output = nn.functional.softmax(self.fc_outerwear(x))
		pants_output = nn.functional.softmax(self.fc_pants(x))
		shoes_output = nn.functional.softmax(self.fc_shoes(x))
		return shirt_output, outerwear_output, pants_output, shoes_output
