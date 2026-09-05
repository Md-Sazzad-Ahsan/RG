import torch
import torch.nn as nn
from torchvision import models

class FundusClassifier(nn.Module):
    def __init__(self, weight_path: str):
        super(FundusClassifier, self).__init__()
        self.model = models.efficientnet_b0(weights=None)
        
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, 2)
        
        self.model.load_state_dict(torch.load(weight_path, map_location="cpu"))
        self.model.eval()

    def predict(self, tensor_image: torch.Tensor) -> bool:
        with torch.no_grad():
            output = self.model(tensor_image)
            probs = torch.softmax(output, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            
            print(f"[Fundus Debug] Output Probs: {probs.tolist()}, Predicted Class: {prediction}")
            
            return prediction == 0
