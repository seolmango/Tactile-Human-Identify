import torch
from tactile.model import CustomModel

class classifier():
    def __init__(self, state_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CustomModel(3).to(self.device)
        self.model.load_state_dict(torch.load(state_path))
        self.model.eval()

    def predict(self, data):
        data = torch.from_numpy(data).float().to(self.device)
        with torch.no_grad():
            output = self.model(data)
            _, predicted = torch.max(output, 1)
        return predicted.item()