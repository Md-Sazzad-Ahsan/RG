from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.core.database import get_db
from app.models.models import Hospital, User, UserRole
from app.core.security import get_current_user_token, get_password_hash

router = APIRouter(prefix="/api/super-admin", tags=["Super Admin"])

# ফ্রন্টএন্ড থেকে আসা ডেটার স্ট্রাকচার
class HospitalCreateRequest(BaseModel):
    hospital_name: str
    hospital_code: str
    hospital_address: str
    admin_name: str
    admin_email: str
    admin_password: str

@router.post("/hospitals")
def create_hospital(
    request: HospitalCreateRequest, 
    db: Session = Depends(get_db), 
    current_user: dict = Depends(get_current_user_token)
):
    # ১. চেক করা ইউজার সত্যিই সুপার এডমিন কি না
    if current_user.get("role") != UserRole.super_admin:
        raise HTTPException(status_code=403, detail="এই কাজটি করার অনুমতি আপনার নেই!")
    
    # ২. চেক করা হসপিটাল কোড বা ইমেইল আগে থেকেই আছে কি না
    if db.query(Hospital).filter(Hospital.code == request.hospital_code).first():
        raise HTTPException(status_code=400, detail="এই হসপিটাল কোডটি আগে থেকেই ব্যবহৃত হচ্ছে।")
    if db.query(User).filter(User.email == request.admin_email).first():
        raise HTTPException(status_code=400, detail="এই ইমেইলটি আগে থেকেই সিস্টেমে আছে।")

    # ৩. নতুন হসপিটাল তৈরি
    new_hospital = Hospital(
        name=request.hospital_name,
        code=request.hospital_code,
        address=request.hospital_address
    )
    db.add(new_hospital)
    db.commit()
    db.refresh(new_hospital) # নতুন হসপিটালের আইডি পাওয়ার জন্য

    # ৪. হসপিটালের জন্য নতুন এডমিন তৈরি
    new_admin = User(
        hospital_id=new_hospital.id,
        full_name=request.admin_name,
        email=request.admin_email,
        password_hash=get_password_hash(request.admin_password),
        role=UserRole.hospital_admin
    )
    db.add(new_admin)
    db.commit()

    return {"message": "হসপিটাল এবং এডমিন সফলভাবে তৈরি হয়েছে!"}


@router.get("/hospitals")
def get_all_hospitals(
    db: Session = Depends(get_db), 
    current_user: dict = Depends(get_current_user_token)
):
    # চেক করা ইউজার সত্যিই সুপার এডমিন কি না
    if current_user.get("role") != UserRole.super_admin:
        raise HTTPException(status_code=403, detail="এই কাজটি করার অনুমতি আপনার নেই!")
    
    # ডেটাবেস থেকে সব হসপিটালের লিস্ট নিয়ে আসা
    hospitals = db.query(Hospital).all()
    return hospitals