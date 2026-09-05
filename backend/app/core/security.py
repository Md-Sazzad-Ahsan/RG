from passlib.context import CryptContext
from jose import jwt
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

# সিক্রেট কি (প্রোডাকশনে এটি .env ফাইলে রাখতে হয়)
SECRET_KEY = "retinaguard_super_secret_key_for_jwt_token"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 120 # টোকেনের মেয়াদ ২ ঘণ্টা

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# পাসওয়ার্ড হ্যাশ করার ফাংশন
def get_password_hash(password: str):
    return pwd_context.hash(password)

# ইউজারের দেওয়া পাসওয়ার্ড এবং ডেটাবেসের হ্যাশ মেলানোর ফাংশন
def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)

# টোকেন জেনারেট করার ফাংশন
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


# টোকেন রিসিভ করার স্কিম
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")

# কারেন্ট ইউজারের ইনফরমেশন বের করার ফাংশন
def get_current_user_token(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("user_id")
        role: str = payload.get("role")
        if user_id is None:
            raise credentials_exception
        return {"user_id": user_id, "role": role}
    except JWTError:
        raise credentials_exception