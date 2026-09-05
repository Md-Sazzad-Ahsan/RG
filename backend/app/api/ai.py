import io
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Depends
from PIL import Image
from sqlalchemy.orm import Session

# Services Import according to project structure
from app.services.dr_service import predict_both_eyes
from app.services.screening_service import (
    generate_patient_id,
    run_full_dr_screening,
    run_single_eye_validation,
)

# আপনার ডাটাবেজ ডিপেন্ডেন্সি এখানে ইমপোর্ট করে নিতে পারেন
# from app.database import get_db

router = APIRouter(prefix="/ai", tags=["AI"])

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg"}


# ============================================================================
# ENDPOINT 1: Single Image Validation (Fundus -> Quality -> Gradability)
# ============================================================================
@router.post("/validate")
async def validate_image(
    file: UploadFile = File(...),
    eye_side: str = Form(..., description="Specify 'left' or 'right'"),
):
    """Validates a single image through:

    1. Fundus Validation
    2. Quality Validation
    3. Gradability Check
    """
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image format for {eye_side} eye. Allowed: JPEG, PNG, JPG.",
        )

    image_bytes = await file.read()

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded file for {eye_side} eye is corrupted or not a valid image.",
        )

    validation_result = await run_single_eye_validation(image)

    if not validation_result["passed"]:
        return {
            "success": False,
            "status": "FAIL",
            "eye_side": eye_side,
            "stage": validation_result.get("failed_stage"),
            "message": validation_result["reason"],
        }

    return {
        "success": True,
        "status": "PASS",
        "eye_side": eye_side,
        "message": "Passed Fundus, Quality, and Gradability checks successfully.",
    }


# ============================================================================
# ENDPOINT 2: Final AI Prediction (DR Model + Grad-CAM Heatmaps + Patient Data)
# ============================================================================
@router.post("/predict")
async def predict(
    left_eye: UploadFile = File(...),
    right_eye: UploadFile = File(...),
    patient_id: str = Form(None),
    hospital_code: str = Form(...),
    full_name: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    phone: str = Form(...),
    address: str = Form(None),
    diabetes_type: str = Form(...),
    diabetes_duration: str = Form(...),
    previous_exam: str = Form(...),
    screening_notes: str = Form(None),
    # db: Session = Depends(get_db)
):
    """Runs final DR Prediction and Grad-CAM generation after validating both

    images, and handles patient ID assignment.
    """
    if left_eye.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Invalid left eye image format.")

    if right_eye.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Invalid right eye image format.")

    left_bytes = await left_eye.read()
    right_bytes = await right_eye.read()

    try:
        left_image = Image.open(io.BytesIO(left_bytes)).convert("RGB")
        right_image = Image.open(io.BytesIO(right_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400, detail="One or both uploaded files are invalid images."
        )

    # Mandatory Validation Check (Both Eyes Must Pass)
    left_val = await run_single_eye_validation(left_image)
    if not left_val["passed"]:
        raise HTTPException(
            status_code=422,
            detail=f"Left Eye Validation Failed ({left_val.get('failed_stage')}): {left_val['reason']}",
        )

    right_val = await run_single_eye_validation(right_image)
    if not right_val["passed"]:
        raise HTTPException(
            status_code=422,
            detail=f"Right Eye Validation Failed ({right_val.get('failed_stage')}): {right_val['reason']}",
        )

    if not patient_id:
        patient_id = f"{hospital_code}-2026-08-001"

    # AI Prediction & Heatmap Generation
    result = predict_both_eyes(left_image, right_image)

    return {
        "success": True,
        "model": "EfficientNet-B7",
        "validation_status": "BOTH_PASSED",
        "patient_info": {
            "patient_id": patient_id,
            "full_name": full_name,
            "age": age,
            "gender": gender,
            "phone": phone,
            "hospital_code": hospital_code,
        },
        "prediction": result,
    }
