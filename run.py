import torch
import torch.nn as nn

# Define your model architecture
class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(17, 1)

    def forward(self, x):
        output = torch.sigmoid(self.linear(x))
        return output


# Load the saved model from file
model = LogisticRegression()
model.load_state_dict(torch.load('/Users/suhaas/Desktop/major_project/SUCCESFUL RUNS/Original Data/03222023_195319_LogisticRegression_30_rounds_0.1LR/Global_Model.pt'))

# Define your input data as a tensor
inputs = torch.randn(1, 17)
inputs = torch.tensor([24,1,1,0,5.0,10.0,1,1,8,5,2,0,1,6.0,0,0,0])
# Make a prediction with the loaded model
output = model(inputs)

# Print the output
print(output)