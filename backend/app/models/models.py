from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, Enum, DateTime, Float
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from app.core.database import Base

class UserRole(str, enum.Enum):
    super_admin = "super_admin"
    hospital_admin = "hospital_admin"
    technician = "technician"
    doctor = "doctor"

class ReportStatus(str, enum.Enum):
    pending = "pending"
    reviewed = "reviewed"

class DRLevel(str, enum.Enum):
    no_dr = "No DR"
    mild = "Mild"
    moderate = "Moderate"
    severe = "Severe"
    proliferative = "Proliferative"

class Hospital(Base):
    __tablename__ = "hospitals"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), index=True)
    code = Column(String(50), unique=True, index=True) # যেমন: DMCH
    address = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)

    users = relationship("User", back_populates="hospital")
    patients = relationship("Patient", back_populates="hospital")

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    hospital_id = Column(Integer, ForeignKey("hospitals.id"), nullable=True)
    full_name = Column(String(100))
    email = Column(String(100), unique=True, index=True)
    password_hash = Column(String(255))
    role = Column(Enum(UserRole))
    is_active = Column(Boolean, default=True)

    hospital = relationship("Hospital", back_populates="users")

class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    hospital_id = Column(Integer, ForeignKey("hospitals.id"))
    patient_id_str = Column(String(50), unique=True, index=True) # যেমন: DMCH-2026-08-001
    name = Column(String(100))
    age = Column(Integer)
    gender = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)

    hospital = relationship("Hospital", back_populates="patients")
    reports = relationship("ScreeningReport", back_populates="patient")

class ScreeningReport(Base):
    __tablename__ = "screening_reports"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    technician_id = Column(Integer, ForeignKey("users.id"))
    doctor_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    left_eye_image = Column(String(255))
    right_eye_image = Column(String(255))
    dr_level = Column(Enum(DRLevel))
    confidence_score = Column(Float)
    
    status = Column(Enum(ReportStatus), default=ReportStatus.pending)
    prescription_notes = Column(String(1000), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    reviewed_at = Column(DateTime, nullable=True)

    patient = relationship("Patient", back_populates="reports")