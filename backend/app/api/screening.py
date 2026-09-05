import io
from datetime import datetime
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Depends
from PIL import Image
from sqlalchemy.orm import Session

# ডাটাবেজ ডিপেন্ডেন্সি ইমপোর্ট
from app.core.database import get_db 
from app.models.models import ScreeningReport, Patient, ReportStatus, DRLevel

# আপনার সার্ভিসগুলো ইমপোর্ট করা হলো
from app.services.dr_service import predict_both_eyes
from app.services.screening_service import (
    generate_patient_id,
    run_full_dr_screening,
    run_single_eye_validation,
)

router = APIRouter(prefix="/screening", tags=["Screening"])

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg"}


# ============================================================================
# 1. Single Image Validation Endpoint (Fundus -> Quality -> Gradability)
# ============================================================================
@router.post("/validate")
async def validate_image(
    file: UploadFile = File(...),
    eye_side: str = Form(..., description="Specify 'left' or 'right'"),
):
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
# 2. Final AI Prediction Endpoint (DR Model + Grad-CAM Heatmaps + Database Save)
# ============================================================================
@router.post("/predict")
async def predict_screening(
    left_eye: UploadFile = File(...),
    right_eye: UploadFile = File(...),
    patient_id: str = Form(None),  
    hospital_code: str = Form("DMCH"),  
    full_name: str = Form(...),
    age: str = Form(...),  
    gender: str = Form(...),
    phone: str = Form(...),
    address: str = Form(None),
    diabetes_type: str = Form("Type 2"),
    diabetes_duration: str = Form("Unknown"),
    previous_exam: str = Form("No"),
    screening_notes: str = Form(None),
    db: Session = Depends(get_db)
):
    try:
        parsed_age = int(age)
    except (ValueError, TypeError):
        parsed_age = 0

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

    # Patient ম্যানেজমেন্ট
    patient_str_id = patient_id if patient_id else f"{hospital_code}-2026-08-001"
    
    db_patient = db.query(Patient).filter(Patient.patient_id_str == patient_str_id).first()
    if not db_patient:
        db_patient = Patient(
            hospital_id=1, 
            patient_id_str=patient_str_id,
            name=full_name,
            age=parsed_age,
            gender=gender
        )
        db.add(db_patient)
        db.commit()
        db.refresh(db_patient)

    # AI Prediction & Heatmap Generation
    result = predict_both_eyes(left_image, right_image)
    
    print("--- DEBUG AI PREDICTION RESULT ---")
    print(result)
    print("----------------------------------")

    # প্রেডিশন থেকে লেবেল বের করা
    final_pred = result.get("final_dr_prediction", {})
    raw_dr_label = (
        final_pred.get("label") or 
        final_pred.get("class_name") or 
        result.get("label") or 
        "No DR"
    )

    # কনফিডেন্স স্কোর সঠিকভাবে বের করার লজিক (যদি final_pred এ না থাকে, তবে ডান বা বাম চোখের হাইয়েস্ট কনফিডেন্স নেওয়া হবে)
    confidence_score_val = float(
        final_pred.get("confidence") or 
        final_pred.get("score") or 
        result.get("right_eye", {}).get("confidence") or 
        result.get("left_eye", {}).get("confidence") or 
        0.0
    )

    # আপডেট করা ফ্লেক্সিবল ম্যাপিং ডিকশনারি (যাতে 'Proliferative DR', 'Mild DR' ইত্যাদি সঠিকভাবে ম্যাপ হয়)
    dr_mapping = {
        "no dr": DRLevel.no_dr,
        "no_dr": DRLevel.no_dr,
        "normal": DRLevel.no_dr,
        "mild": DRLevel.mild,
        "mild dr": DRLevel.mild,
        "moderate": DRLevel.moderate,
        "moderate dr": DRLevel.moderate,
        "severe": DRLevel.severe,
        "severe dr": DRLevel.severe,
        "proliferative": DRLevel.proliferative,
        "proliferative dr": DRLevel.proliferative
    }
    
    normalized_label = str(raw_dr_label).strip().lower()
    dr_enum_val = dr_mapping.get(normalized_label, DRLevel.no_dr)

    # Database Saving
    try:
        new_report = ScreeningReport(
            patient_id=db_patient.id,
            technician_id=1, 
            left_eye_image=left_eye.filename,
            right_eye_image=right_eye.filename,
            dr_level=dr_enum_val,
            confidence_score=confidence_score_val,
            status=ReportStatus.pending,
            prescription_notes=screening_notes,
            created_at=datetime.utcnow()
        )
        db.add(new_report)
        db.commit()
        db.refresh(new_report)
        
        print(f"Screening report saved for: {full_name} | Detected Level: {raw_dr_label} | Score: {confidence_score_val}")

    except Exception as db_err:
        db.rollback()
        print("Database save error:", str(db_err))
        raise HTTPException(status_code=500, detail=f"Database error: {str(db_err)}")

    return {
        "success": True,
        "model": "EfficientNet-B7",
        "validation_status": "BOTH_PASSED",
        "patient_info": {
            "patient_id": db_patient.patient_id_str,
            "full_name": full_name,
            "age": parsed_age,
            "gender": gender,
            "phone": phone,
            "hospital_code": hospital_code,
            "screening_notes": screening_notes,
        },
        "prediction": result,
    }