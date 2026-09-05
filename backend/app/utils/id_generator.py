from datetime import datetime

def generate_patient_id(hospital_code: str, last_serial: int) -> str:
    """
    Generate dynamic patient ID format: [HOSPITAL_CODE]-[YYYY]-[MM]-[SERIAL]
    """
    now = datetime.now()
    year = now.strftime("%Y")
    month = now.strftime("%m")
    
    # নতুন সিরিয়ালটি আগের সিরিয়ালের সাথে ১ যোগ করে ৩ ডিজিট ফরম্যাটে হবে
    new_serial = f"{(last_serial + 1):03d}"
    
    patient_id = f"{hospital_code}-{year}-{month}-{new_serial}"
    return patient_id