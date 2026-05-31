# data-pipeline/database.py
import mysql.connector

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "07102005", 
    "database": "job_recommendation_db"
}

def init_db():
    conn = mysql.connector.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            job_id VARCHAR(50) PRIMARY KEY,
            source VARCHAR(50),
            job_title VARCHAR(255),
            company_name VARCHAR(255),
            job_location VARCHAR(100),
            job_type VARCHAR(50),
            category VARCHAR(100),
            salary_min DECIMAL(10,2),
            salary_max DECIMAL(10,2),
            salary_avg DECIMAL(10,2),
            salary_range VARCHAR(100),
            salary_currency VARCHAR(10),
            experience_required INT,
            education_required VARCHAR(100),
            skills_required TEXT,
            job_description TEXT,
            link_url TEXT,
            
            -- BONUS HIGH-VALUE FIELDS FOR NO-EDUCATION APPLICANTS
            gender_preference VARCHAR(20),
            contact_telegram VARCHAR(100),
            contact_phone VARCHAR(50),
            contact_email VARCHAR(255),
            
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

def save_job_to_db(job: dict):
    conn = mysql.connector.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    query = """
        INSERT INTO jobs (
            job_id, source, job_title, company_name, job_location, job_type, category,
            salary_min, salary_max, salary_avg, salary_range, salary_currency,
            experience_required, education_required, skills_required, job_description, link_url,
            gender_preference, contact_telegram, contact_phone, contact_email
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE 
            job_title = VALUES(job_title),
            salary_range = VALUES(salary_range),
            salary_min = VALUES(salary_min),
            salary_max = VALUES(salary_max),
            salary_avg = VALUES(salary_avg),
            experience_required = VALUES(experience_required),
            job_description = VALUES(job_description),
            contact_telegram = VALUES(contact_telegram),
            contact_phone = VALUES(contact_phone);
    """
    
    cur.execute(query, (
        job["job_id"], job["source"], job["job_title"], job["company_name"], job["job_location"],
        job["job_type"], job["category"], job["salary_min"], job["salary_max"], job["salary_avg"],
        job["salary_range"], job["salary_currency"], job["experience_required"], job["education_required"],
        job["skills_required"], job["job_description"], job["link_url"],
        job["gender_preference"], job["contact_telegram"], job["contact_phone"], job["contact_email"]
    ))
    
    conn.commit()
    cur.close()
    conn.close()