from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms
from app.models.gradability_model import GradabilityClassifier

# Base path resolve for weights (Updated according to project structure)
BASE_DIR = Path(__file__).resolve().parents[2]
WEIGHTS_PATH = BASE_DIR / "app" / "ml_pipeline" / "models" / "gradability_best.pth"

# Transform for Gradability Check Model
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Model Initialization
gradability_classifier = GradabilityClassifier(str(WEIGHTS_PATH))


async def validate_gradability(image: Image.Image) -> dict:
    """
    Validates whether anatomical structures (optic disc, blood vessels) are readable by the AI model.
    """
    try:
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        input_tensor = transform(image).unsqueeze(0)
        
        is_gradable = gradability_classifier.predict(input_tensor)
        
        if not is_gradable:
            return {
                "passed": False, 
                "reason": "Image is ungradable (key anatomical structures are obstructed or missing)."
            }
            
        return {
            "passed": True, 
            "reason": "Passed Gradability Check."
        }

    except Exception as e:
        return {
            "passed": False, 
            "reason": f"Gradability check error: {str(e)}"
        }
