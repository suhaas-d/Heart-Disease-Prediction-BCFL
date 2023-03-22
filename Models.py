import torch
import torch.nn as nn
import torch.nn.functional as F


class Mnist_2NN(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(784, 200)
		self.fc2 = nn.Linear(200, 200)
		self.fc3 = nn.Linear(200, 10)

	def forward(self, inputs):
		tensor = F.relu(self.fc1(inputs))
		tensor = F.relu(self.fc2(tensor))
		tensor = self.fc3(tensor)
		return tensor


class Mnist_CNN(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
		self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
		self.fc1 = nn.Linear(7*7*64, 512)
		self.fc2 = nn.Linear(512, 10)

	def forward(self, inputs):
		tensor = inputs.view(-1, 1, 28, 28)
		tensor = F.relu(self.conv1(tensor))
		tensor = self.pool1(tensor)
		tensor = F.relu(self.conv2(tensor))
		tensor = self.pool2(tensor)
		tensor = tensor.view(-1, 7*7*64)
		tensor = F.relu(self.fc1(tensor))
		tensor = self.fc2(tensor)
		return tensor

class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(17, 1)

    def forward(self, x):
        output = torch.sigmoid(self.linear(x))
        return output

class BRFSS_NN(nn.Module):
    def __init__(self):
        super(BRFSS_NN, self).__init__()
        self.layer1 = nn.Linear(17, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return torch.sigmoid(x)