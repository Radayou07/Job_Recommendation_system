# data-pipeline/cleaner.py
import re

def parse_salary_string(salary_str: str):
    """
    Parses strings like '$400 - $600' or '400$ - 600$' or 'Negotiable'
    Returns: (min, max, avg, currency)
    """
    if not salary_str or "negotiable" in salary_str.lower():
        return None, None, None, "USD" # Default currency in KH job boards is usually USD
        
    # Extract all numbers from the string using regex
    numbers = [float(n) for n in re.findall(r'\d+', salary_str)]
    
    if len(numbers) >= 2:
        sal_min = numbers[0]
        sal_max = numbers[1]
        sal_avg = (sal_min + sal_max) / 2
        return sal_min, sal_max, sal_avg, "USD"
    elif len(numbers) == 1:
        return numbers[0], numbers[0], numbers[0], "USD"
        
    return None, None, None, "USD"

def clean_job_payload(raw_data: dict) -> dict:
    if not raw_data:
        return None
        
    # Get nested employer values safely
    employer = raw_data.get("employer", {})
    contact = raw_data.get("contact", {})
    
    # Extract locations array safely
    locations = raw_data.get("locations", [])
    location_label = "Phnom Penh"
    if locations and isinstance(locations, list):
        location_label = locations[0].get("locationId", {}).get("label", "Phnom Penh")

    # Handle the salary calculations
    raw_salary_range = raw_data.get("salaryId", {}).get("label", "Negotiable").strip()
    s_min, s_max, s_avg, s_curr = parse_salary_string(raw_salary_range)

    # Compile explicit languages as skill requirements if present
    langs = [l.get("languageId", {}).get("label") for l in raw_data.get("jobLangs", []) if l.get("languageId")]
    skills_list = ", ".join(langs) if langs else "None Specified"

    # Social helper checks: look for Telegram markers
    telegram_handle = None
    online_apply = raw_data.get("onlineApply", "")
    if online_apply and "@" in online_apply:
        telegram_handle = online_apply.strip()

    return {
        "job_id": raw_data.get("id"),
        "source": "CamHR",
        "job_title": raw_data.get("title", "Unknown Title").strip(),
        "company_name": employer.get("company", "Unknown Company").strip(),
        "job_location": location_label.strip(),
        "job_type": raw_data.get("termId", {}).get("label", "Full Time").strip(),
        "category": raw_data.get("categoryId", {}).get("label", "General").strip(),
        
        # Parsed Salaries
        "salary_min": s_min,
        "salary_max": s_max,
        "salary_avg": s_avg,
        "salary_range": raw_salary_range,
        "salary_currency": s_curr,
        
        # Qualifications
        "experience_required": raw_data.get("workyears", 0), # 0 means fresh/no experience!
        "education_required": raw_data.get("qualificationId", {}).get("label", "No Degree Required").strip() or "No Degree Required",
        "skills_required": skills_list,
        
        # Explanatory text blocks
        "job_description": f"REQUIREMENTS:\n{raw_data.get('requirement', '')}\n\nDESCRIPTION:\n{raw_data.get('description', '')}",
        "link_url": f"https://camhr.com/a/job/{raw_data.get('id')}",
        
        # CRITICAL CONTACT AND BIAS FIELDS FOR ACCESSIBILITY MATCHING
        "gender_preference": raw_data.get("sex", {}).get("label", "Any").strip(),
        "contact_telegram": telegram_handle,
        "contact_phone": contact.get("telephone", "").strip(),
        "contact_email": contact.get("email", employer.get("email", "")).strip()
    }