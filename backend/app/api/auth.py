from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.models import User
from app.core.security import verify_password, create_access_token
from pydantic import BaseModel
from app.core.security import verify_password, create_access_token, get_password_hash
from app.models.models import User, UserRole

router = APIRouter(prefix="/api/auth", tags=["Authentication"])

# ফ্রন্টএন্ড থেকে আসা ডেটার স্ট্রাকচার
class LoginRequest(BaseModel):
    email: str
    password: str

@router.post("/login")
def login(request: LoginRequest, db: Session = Depends(get_db)):
    # ১. ডেটাবেস থেকে ইমেইল দিয়ে ইউজার খোঁজা
    user = db.query(User).filter(User.email == request.email).first()
    
    # ২. ইউজার না থাকলে বা পাসওয়ার্ড ভুল হলে এরর দেওয়া
    if not user or not verify_password(request.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="ভুল ইমেইল বা পাসওয়ার্ড",
        )
    
    # ৩. সব ঠিক থাকলে ইউজারের ইনফরমেশন দিয়ে টোকেন তৈরি করা
    token_data = {
        "user_id": user.id,
        "role": user.role,
        "hospital_id": user.hospital_id
    }
    access_token = create_access_token(data=token_data)

    # ৪. টোকেন এবং ইউজারের রোল ফ্রন্টএন্ডে পাঠানো
    return {
        "access_token": access_token, 
        "token_type": "bearer",
        "role": user.role,
        "full_name": user.full_name
    }

@router.post("/setup-super-admin")
def setup_super_admin(db: Session = Depends(get_db)):
    # চেক করব সুপার এডমিন আগে থেকেই আছে কি না
    existing_admin = db.query(User).filter(User.email == "admin@retinaguard.com").first()
    if existing_admin:
        return {"message": "সুপার এডমিন আগে থেকেই ডেটাবেসে আছে!"}

    # নতুন সুপার এডমিন তৈরি
    hashed_pwd = get_password_hash("admin123") # ডিফল্ট পাসওয়ার্ড
    new_admin = User(
        full_name="Sinthol Dey",
        email="admin@retinaguard.com",
        password_hash=hashed_pwd,
        role=UserRole.super_admin
    )
    db.add(new_admin)
    db.commit()
    return {"message": "সুপার এডমিন সফলভাবে তৈরি হয়েছে! ইমেইল: admin@retinaguard.com, পাসওয়ার্ড: admin123"}