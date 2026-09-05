import torch
import torch.nn as nn
from torchvision import models

class QualityClassifier(nn.Module):
    def __init__(self, weight_path: str):
        super(QualityClassifier, self).__init__()
        self.model = models.efficientnet_b0(weights=None)
        
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, 3) # আপনার মডেলটিতে ৩টি আউটপুট লেয়ার ছিল
        
        try:
            state_dict = torch.load(weight_path, map_location="cpu")
            self.model.load_state_dict(state_dict)
        except Exception as e:
            print(f"[Warning] Could not load quality model weights: {e}")
            
        self.model.eval()

    def predict(self, tensor_image: torch.Tensor) -> bool:
        with torch.no_grad():
            outputs = self.model(tensor_image)
            probs = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            
            print(f"[Quality Debug] Output Probs: {probs.tolist()}, Predicted Class: {prediction}")
            
            return prediction == 0
