import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(3, 16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

if __name__ == "__main__":
    model = SimpleModel()
    torch.save(model.state_dict(), "../models/trained_model.pth")
    print("Model saved successfully.")
