from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# XAMPP ব্যবহার করলে সাধারণত ইউজারনেম root এবং পাসওয়ার্ড ফাঁকা থাকে
SQLALCHEMY_DATABASE_URL = "mysql+pymysql://root:12345@localhost:3306/retinaguard_db"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ডাটাবেস সেশন পাওয়ার জন্য ডিপেন্ডেন্সি ফাংশন
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()