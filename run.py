import torch
import torch.nn as nn
import pickle
# Define your model architecture
class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(17, 1)

    def forward(self, x):
        output = torch.sigmoid(self.linear(x))
        return output
class BinaryClassification(nn.Module):
    def __init__(self):
        super(BinaryClassification, self).__init__()
        # Number of input features is 12.
        self.layer_1 = nn.Linear(17, 64) 
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 1) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        
        return x
with open('/Users/suhaas/Desktop/major_project/Heart_Disease_Prediction_BCFL/SUCCESFUL RUNS/Oversampled Data/03232023_231818_BinaryClassification_10_rounds_0.1LR/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
# Load the saved model from file
model = BinaryClassification()
model.load_state_dict(torch.load('/Users/suhaas/Desktop/major_project/Heart_Disease_Prediction_BCFL/SUCCESFUL RUNS/Oversampled Data/03232023_231818_BinaryClassification_10_rounds_0.1LR/Global_Model.pt'))


# Define your input data as a tensor
inputs = torch.randn(1, 17)
# print(inputs.dtype)
inputs = torch.tensor([27.038640227480787,1,0,0,0.0,0.0,0,1,7,3,0,1,1,7.49150142175492,0,0,0])
inputs = inputs.reshape(1,-1)
inputs = scaler.transform(inputs)
print(inputs)
inputs = torch.tensor(inputs.reshape(1,17))
inputs = inputs.to(torch.float32)
# Make a prediction with the loaded model
model.eval()
output = model(inputs)

# Print the output
# print(output)

print(torch.sigmoid(output))