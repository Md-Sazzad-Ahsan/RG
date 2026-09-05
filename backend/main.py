from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.database import engine, Base
from app.models import models
from app.api import auth, super_admin, hospital_admin, screening
from fastapi.staticfiles import StaticFiles
import os

# ডাটাবেসে টেবিলগুলো তৈরি করার কমান্ড
Base.metadata.create_all(bind=engine)

app = FastAPI(title="RetinaGuard API", description="Backend API for RetinaGuard SaaS")

# CORS সেটআপ (এটি ছাড়া ফ্রন্টএন্ড থেকে এপিআই কল ব্রাউজার ব্লক করে দেবে)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # লোকালহোস্ট থেকে রিকোয়েস্ট অ্যালাউ করার জন্য
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- হিটম্যাপ এবং স্ট্যাটিক ইমেজ ফাইল সার্ভ করার জন্য মাউন্টিং ---
# (নিশ্চিত করুন আপনার প্রজেক্ট রুটে outputs বা heatmaps ফোল্ডারটি এই পাথে রয়েছে)
os.makedirs("outputs/heatmaps", exist_ok=True)
app.mount("/heatmaps", StaticFiles(directory="outputs/heatmaps"), name="heatmaps")

@app.get("/")
def read_root():
    return {"message": "Welcome to RetinaGuard API. The server is running successfully!"}

# রাউটগুলো যুক্ত করা হলো
app.include_router(auth.router)
app.include_router(super_admin.router)
app.include_router(hospital_admin.router)
app.include_router(screening.router)