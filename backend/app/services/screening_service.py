from datetime import datetime
from PIL import Image
from fastapi import HTTPException
from sqlalchemy.orm import Session

# প্রজেক্ট ফোল্ডার স্ট্রাকচার অনুযায়ী সার্ভিসগুলো ইমপোর্ট করা হয়েছে
from app.services.fundus_service import validate_fundus
from app.services.quality_service import validate_quality
from app.services.gradability_service import validate_gradability
from app.services.dr_service import predict_single_eye, predict_both_eyes


async def run_single_eye_validation(image: Image.Image) -> dict:
    """Runs an image sequentially through:

    1. Fundus Validation
    2. Quality Check
    3. Gradability Check
    """
    # 1. Fundus Check
    fundus_res = await validate_fundus(image)
    if not fundus_res.get("passed", False):
        return {
            "passed": False,
            "failed_stage": "Fundus Validation",
            "reason": fundus_res.get("reason", "Invalid Fundus Image"),
        }

    # 2. Quality Check
    quality_res = await validate_quality(image)
    if not quality_res.get("passed", False):
        return {
            "passed": False,
            "failed_stage": "Quality Validation",
            "reason": quality_res.get("reason", "Poor Image Quality"),
        }

    # 3. Gradability Check
    grad_res = await validate_gradability(image)
    if not grad_res.get("passed", False):
        return {
            "passed": False,
            "failed_stage": "Gradability Check",
            "reason": grad_res.get("reason", "Ungradable Image"),
        }

    return {"passed": True}


async def run_full_dr_screening(image: Image.Image) -> dict:
    """Validates single eye and performs DR Prediction using predict_single_eye."""
    validation_res = await run_single_eye_validation(image)
    if not validation_res["passed"]:
        return validation_res

    dr_result = predict_single_eye(image)
    return {"passed": True, "dr_result": dr_result}


# অটো-আইডি জেনারেটর ফাংশন
def generate_patient_id(hospital_code: str, db: Session, PatientModel) -> str:
    now = datetime.now()
    year = now.strftime("%Y")
    month = now.strftime("%m")

    prefix = f"{hospital_code}-{year}-{month}-"

    last_patient = (
        db.query(PatientModel)
        .filter(PatientModel.patient_id.like(f"{prefix}%"))
        .order_by(PatientModel.id.desc())
        .first()
    )

    if last_patient:
        last_serial_str = last_patient.patient_id.split("-")[-1]
        new_serial = int(last_serial_str) + 1
    else:
        new_serial = 1

    return f"{prefix}{new_serial:03d}"
