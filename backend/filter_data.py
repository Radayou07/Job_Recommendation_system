import pandas as pd
import os
import re

# Paths
LARGE_DATA_PATH = "/media/radayou/DATA/Study/ITC/Year 3/2nd semester/Mini Project/Final/Job_Recommendation/model/job_descriptions.csv"
OUTPUT_PATH = "/media/radayou/DATA/Study/ITC/Year 3/2nd semester/Mini Project/Final/Job_Recommendation/backend/data/data.csv"

def parse_salary(salary_str):
    """Parses '$64K-$92K' to (64000, 92000, 78000, '$64K-$92K')"""
    try:
        nums = re.findall(r'\d+', salary_str.replace('K', '000'))
        if len(nums) >= 2:
            s_min, s_max = float(nums[0]), float(nums[1])
            return s_min, s_max, (s_min + s_max) / 2, salary_str
    except:
        pass
    return 0, 0, 0, salary_str

def parse_experience(exp_str):
    """Parses '5 to 14 Years' to 5"""
    try:
        nums = re.findall(r'\d+', str(exp_str))
        if nums:
            return int(nums[0])
    except:
        pass
    return 0

def process_data():
    print("Starting data filtering and merging...")
    
    # Check if target file exists to preserve it
    if os.path.exists(OUTPUT_PATH):
        df_original = pd.read_csv(OUTPUT_PATH)
        print(f"Original data.csv has {len(df_original)} rows.")
    else:
        df_original = pd.DataFrame()
        print("Creating new data.csv.")

    # Process large file in chunks
    chunk_size = 50000
    relevant_chunks = []
    
    # Mapping for large dataset columns to data.csv columns
    # source: Job Portal
    # job_title: Job Title
    # company_name: Company
    # job_location: location
    # job_type: Work Type
    # category: Role
    # experience_required: Experience (parsed)
    # education_required: Qualifications
    # skills_required: skills
    # job_description: Job Description + Responsibilities
    
    try:
        for chunk in pd.read_csv(LARGE_DATA_PATH, chunksize=chunk_size):
            # 1. Filter for Cambodia
            cambodia_jobs = chunk[chunk['Country'] == 'Cambodia'].copy()
            
            if not cambodia_jobs.empty:
                # 2. Filter for low education (Optional but recommended based on goal)
                # The user goal says "low-education workers"
                low_edu_keywords = ['High School', 'Secondary School', 'Primary', 'None', 'Vocational', 'Diploma']
                edu_mask = cambodia_jobs['Qualifications'].str.contains('|'.join(low_edu_keywords), case=False, na=False)
                # We also keep some "Bachelor" just in case, or just take all Cambodia jobs as requested?
                # User said: "it will be Cambodia's job only"
                # Let's take all Cambodia jobs to maximize accuracy/data pool.
                
                # 3. Map columns
                mapped = pd.DataFrame()
                mapped['source'] = cambodia_jobs['Job Portal']
                mapped['job_title'] = cambodia_jobs['Job Title']
                mapped['company_name'] = cambodia_jobs['Company']
                mapped['job_location'] = cambodia_jobs['location']
                mapped['job_type'] = cambodia_jobs['Work Type']
                mapped['category'] = cambodia_jobs['Role']
                
                # Salary parsing
                salaries = cambodia_jobs['Salary Range'].apply(parse_salary)
                mapped['salary_min'] = [s[0] for s in salaries]
                mapped['salary_max'] = [s[1] for s in salaries]
                mapped['salary_avg'] = [s[2] for s in salaries]
                mapped['salary_range'] = [s[3] for s in salaries]
                mapped['salary_currency'] = 'USD'
                
                mapped['experience_required'] = cambodia_jobs['Experience'].apply(parse_experience)
                mapped['education_required'] = cambodia_jobs['Qualifications']
                mapped['skills_required'] = cambodia_jobs['skills']
                mapped['job_description'] = cambodia_jobs['Job Description'] + " Responsibilities: " + cambodia_jobs['Responsibilities']
                
                relevant_chunks.append(mapped)
                print(f"Found {len(mapped)} Cambodia jobs in chunk.")

        if relevant_chunks:
            df_new = pd.concat(relevant_chunks)
            # Combine with original
            df_final = pd.concat([df_original, df_new]).drop_duplicates(subset=['job_title', 'company_name', 'job_description'])
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
            df_final.to_csv(OUTPUT_PATH, index=False)
            print(f"Successfully updated {OUTPUT_PATH}. New total rows: {len(df_final)}")
        else:
            print("No Cambodia jobs found in the large dataset.")
            
    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    process_data()
