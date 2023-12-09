import torch
import torch.nn as nn
import pickle
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
#model.load_state_dict(torch.load('/Users/suhaas/Desktop/major_project/Heart_Disease_Prediction_BCFL/SUCCESFUL RUNS/Oversampled Data/03232023_231818_BinaryClassification_10_rounds_0.1LR/Global_Model.pt'))
X_test = np.asarray(pd.read_csv('/Users/suhaas/Desktop/major_project/Oversampled Data/X_test.csv'))
y_test = np.asarray(pd.read_csv('/Users/suhaas/Desktop/major_project/Oversampled Data/y_test.csv'))
X_test=scaler.fit_transform(X_test)
y_test=torch.from_numpy(y_test.astype(np.float32))
X_test=torch.from_numpy(X_test.astype(np.float32))

# Define your input data as a tensor
inputs = torch.randn(1, 17)
# print(inputs.dtype)
# BMI,Smoking,AlcoholDrinking,Stroke,PhysicalHealth,MentalHealth,DiffWalking,Sex,AgeCategory,Race,Diabetic,PhysicalActivity,GenHealth,SleepTime,Asthma,KidneyDisease,SkinCancer
inputs = torch.tensor([25.84,1,0,0,0.0,0.0,0,1,8,2,0,0,4,3.0,1,0,0])
inputs = inputs.reshape(1,-1)
inputs = scaler.transform(inputs)
print(inputs)
inputs = torch.tensor(inputs.reshape(1,17))
inputs = inputs.to(torch.float32)
# Make a prediction with the loaded model
model.eval()
output = model(X_test)

# Print the output
# print(output)

# print(torch.sigmoid(output))
print('Accuracy:', accuracy_score(y_test, torch.round(torch.sigmoid(output)).detach().numpy()))
print('Precision:', precision_score(y_test, torch.round(torch.sigmoid(output)).detach().numpy()))
print('Recall:', recall_score(y_test, torch.round(torch.sigmoid(output)).detach().numpy()))
print('F1 Score:', f1_score(y_test, torch.round(torch.sigmoid(output)).detach().numpy()))