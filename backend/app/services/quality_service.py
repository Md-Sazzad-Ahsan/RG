from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms
from app.models.quality_model import QualityClassifier

# Base path resolve for weights (Updated according to project structure)
BASE_DIR = Path(__file__).resolve().parents[2]
WEIGHTS_PATH = BASE_DIR / "app" / "ml_pipeline" / "models" / "iqa_best.pth"

# Transform for Quality Validation Model
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Model Initialization
quality_classifier = QualityClassifier(str(WEIGHTS_PATH))


async def validate_quality(image: Image.Image) -> dict:
    """
    Validates whether the fundus image has acceptable quality (blur, lighting, exposure).
    """
    try:
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        input_tensor = transform(image).unsqueeze(0)
        
        is_good_quality = quality_classifier.predict(input_tensor)
        
        if not is_good_quality:
            return {
                "passed": False, 
                "reason": "Image quality is poor (excessive blur, bad lighting, or severe exposure issues)."
            }
            
        return {
            "passed": True, 
            "reason": "Passed Quality Validation."
        }

    except Exception as e:
        return {
            "passed": False, 
            "reason": f"Quality validation error: {str(e)}"
        }
