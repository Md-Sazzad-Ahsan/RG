from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms
from app.models.fundus_model import FundusClassifier

# Base path resolve for weights (Updated according to project structure)
BASE_DIR = Path(__file__).resolve().parents[2]
WEIGHTS_PATH = BASE_DIR / "app" / "ml_pipeline" / "models" / "fundus_best.pth"

# Transform for Fundus Validation Model
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Model Initialization
fundus_classifier = FundusClassifier(str(WEIGHTS_PATH))


async def validate_fundus(image: Image.Image) -> dict:
    """
    Validates whether the provided PIL image is a genuine eye fundus image.
    """
    try:
        # Ensure image is in RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Apply transforms and add batch dimension
        input_tensor = transform(image).unsqueeze(0)
        
        # Run inference
        is_valid = fundus_classifier.predict(input_tensor)
        
        if not is_valid:
            return {
                "passed": False, 
                "reason": "The uploaded image is NOT a valid fundus image."
            }
            
        return {
            "passed": True, 
            "reason": "Valid Fundus Image."
        }

    except Exception as e:
        return {
            "passed": False, 
            "reason": f"Fundus validation error: {str(e)}"
        }
