from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.core.database import get_db
from app.models.models import User, UserRole
from app.core.security import get_current_user_token, get_password_hash

router = APIRouter(prefix="/api/hospital-admin", tags=["Hospital Admin"])

class TechnicianCreate(BaseModel):
    full_name: str
    email: str
    password: str

@router.post("/technicians")
def create_technician(
    req: TechnicianCreate, 
    db: Session = Depends(get_db), 
    current_user: dict = Depends(get_current_user_token)
):
    if current_user.get("role") != UserRole.hospital_admin:
        raise HTTPException(status_code=403, detail="অনুমতি নেই!")
    
    # বর্তমান এডমিন কোন হসপিটালের তা বের করা
    admin = db.query(User).filter(User.id == current_user["user_id"]).first()
    
    if db.query(User).filter(User.email == req.email).first():
        raise HTTPException(status_code=400, detail="এই ইমেইলটি আগে থেকেই ব্যবহৃত হচ্ছে।")

    # নতুন টেকনিশিয়ান তৈরি (admin.hospital_id ব্যবহার করে)
    new_tech = User(
        full_name=req.full_name,
        email=req.email,
        password_hash=get_password_hash(req.password),
        role=UserRole.technician,
        hospital_id=admin.hospital_id
    )
    db.add(new_tech)
    db.commit()
    
    return {"message": "টেকনিশিয়ান সফলভাবে যুক্ত হয়েছে!"}

@router.get("/technicians")
def get_technicians(
    db: Session = Depends(get_db), 
    current_user: dict = Depends(get_current_user_token)
):
    if current_user.get("role") != UserRole.hospital_admin:
        raise HTTPException(status_code=403, detail="অনুমতি নেই!")
        
    admin = db.query(User).filter(User.id == current_user["user_id"]).first()
    
    # শুধুমাত্র এই হসপিটালের টেকনিশিয়ানদের ডেটা পাঠানো হবে
    techs = db.query(User).filter(User.hospital_id == admin.hospital_id, User.role == UserRole.technician).all()
    return techs