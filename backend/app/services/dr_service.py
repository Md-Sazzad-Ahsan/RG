from pathlib import Path
import uuid

import cv2
import numpy as np

from PIL import Image

from torchvision import transforms

from app.models.dr_model import DRModel


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

# আপনার ফোল্ডার স্ট্রাকচার অনুযায়ী ওয়েট ফাইলের সঠিক পাথ
WEIGHTS_PATH = (
    BASE_DIR
    / "app"
    / "ml_pipeline"
    / "models"
    / "efficientnet_b7_best.pth"
)

HEATMAP_DIR = (
    BASE_DIR
    / "outputs"
    / "heatmaps"
)

HEATMAP_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MODEL
# ============================================================

dr_model = DRModel(
    WEIGHTS_PATH
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

preprocess = transforms.Compose([
    transforms.Resize((380, 380)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# DR CLASSES
# ============================================================

DR_CLASSES = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR"
}


# ============================================================
# SINGLE EYE PREDICTION + GRAD-CAM
# ============================================================

def predict_single_eye(
    image: Image.Image
):
    """
    Performs DR prediction and Grad-CAM generation
    for a single fundus image.
    """

    # --------------------------------------------------------
    # Convert image to RGB
    # --------------------------------------------------------

    image = image.convert("RGB")


    # --------------------------------------------------------
    # Create OpenCV version of original image
    #
    # PIL uses RGB
    # OpenCV uses BGR
    # --------------------------------------------------------

    original_image = np.array(
        image
    )

    original_image = cv2.cvtColor(
        original_image,
        cv2.COLOR_RGB2BGR
    )


    # --------------------------------------------------------
    # Preprocess image for EfficientNet-B7
    # --------------------------------------------------------

    input_tensor = preprocess(
        image
    )

    input_tensor = input_tensor.unsqueeze(
        0
    )


    # --------------------------------------------------------
    # Prediction + Grad-CAM
    # --------------------------------------------------------

    result = dr_model.predict_with_gradcam(
        input_tensor,
        original_image
    )


    # --------------------------------------------------------
    # Prediction information
    # --------------------------------------------------------

    class_index = result[
        "class_index"
    ]

    confidence = result[
        "confidence"
    ]

    probabilities = result[
        "probabilities"
    ]

    heatmap = result[
        "heatmap"
    ]


    # --------------------------------------------------------
    # Generate unique heatmap filename
    # --------------------------------------------------------

    heatmap_filename = (
        f"heatmap_{uuid.uuid4().hex}.jpg"
    )


    # --------------------------------------------------------
    # Heatmap save path
    # --------------------------------------------------------

    heatmap_path = (
        HEATMAP_DIR
        / heatmap_filename
    )


    # --------------------------------------------------------
    # Save heatmap image
    # --------------------------------------------------------

    success = cv2.imwrite(
        str(heatmap_path),
        heatmap
    )

    if not success:
        raise RuntimeError(
            "Failed to save Grad-CAM heatmap."
        )


    # --------------------------------------------------------
    # Browser-accessible heatmap URL
    # --------------------------------------------------------

    heatmap_url = (
        f"/heatmaps/{heatmap_filename}"
    )


    # --------------------------------------------------------
    # Return single-eye result
    # --------------------------------------------------------

    return {

        "class_index": class_index,

        "label": DR_CLASSES[
            class_index
        ],

        "confidence": round(
            confidence,
            4
        ),

        "probabilities": [
            round(
                value,
                4
            )
            for value in probabilities
        ],

        "heatmap_url": heatmap_url
    }


# ============================================================
# BOTH EYES PREDICTION
# ============================================================

def predict_both_eyes(
    left_image: Image.Image,
    right_image: Image.Image
):
    """
    Performs DR prediction for both eyes.

    Final DR prediction is determined by the
    maximum DR severity between the two eyes.

    Severity order:

    0 → No DR
    1 → Mild NPDR
    2 → Moderate NPDR
    3 → Severe NPDR
    4 → Proliferative DR
    """

    # --------------------------------------------------------
    # LEFT EYE
    # --------------------------------------------------------

    left_result = predict_single_eye(
        left_image
    )


    # --------------------------------------------------------
    # RIGHT EYE
    # --------------------------------------------------------

    right_result = predict_single_eye(
        right_image
    )


    # --------------------------------------------------------
    # FINAL DR PREDICTION
    #
    # Maximum severity between both eyes
    # --------------------------------------------------------

    final_class = max(
        left_result["class_index"],
        right_result["class_index"]
    )


    final_label = DR_CLASSES[
        final_class
    ]


    # --------------------------------------------------------
    # FINAL RESPONSE
    # --------------------------------------------------------

    return {

        "left_eye": left_result,
        "right_eye": right_result,
        "final_dr_prediction": {
            "class_index": final_class,
            "label": final_label

        }
    }