import torch
from tactile.model import CustomModel
import numpy as np

class Classifier():
    def __init__(self, state_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CustomModel(6).to(self.device)
        self.model.load_state_dict(torch.load(state_path))
        self.model.eval()

    def predict(self, data):
        data = data.reshape(-1, 1, 25, 25)
        data = (data - np.min(data))/(np.max(data)-np.min(data))
        data = torch.from_numpy(data).float().to(self.device)
        with torch.no_grad():
            output = self.model(data)
            _, predicted = torch.max(output, 1)
            # if max value is too similar to each other, return 6
            sorted_output = torch.sort(output, descending=True).values
            if sorted_output[0][0] - sorted_output[0][1] < 3:
                return 6
            else:
                return predicted.item()