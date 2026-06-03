from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from collections import Counter
import pandas as pd
import numpy as np
import torch
import math
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os

app = FastAPI(title="littlehelp API")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration & Model Loading ---
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "data.csv")
MODEL_NAME = "BAAI/bge-base-en-v1.5"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Coordinate map for Cambodian Provinces
COORDS_MAP = {
    "Phnom Penh": (11.5564, 104.9282),
    "Kandal": (11.4746, 104.9474),
    "Siem Reap": (13.3671, 103.8448),
    "Sihanoukville": (10.6093, 103.5296),
    "Battambang": (13.0957, 103.2022),
    "Kampong Cham": (11.9924, 105.4645),
    "Kampot": (10.5942, 104.1814),
    "Kratié": (12.4881, 106.0187),
    "Mondulkiri": (12.4558, 107.1747),
    "Preah Vihear": (13.8073, 104.9811),
    "Ratanakiri": (13.8577, 107.0125),
    "Takeo": (10.9908, 104.7846),
    "Remote": None
}

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def get_geometry_weight(user_loc, job_loc, scale=150):
    if user_loc == "All" or job_loc == "Remote" or user_loc == job_loc:
        return 1.0
    p1, p2 = COORDS_MAP.get(user_loc), COORDS_MAP.get(job_loc)
    if not p1 or not p2:
        return 0.5
    dist = haversine_distance(p1[0], p1[1], p2[0], p2[1])
    return math.exp(-dist / scale)

# Global variables
df = None
bow_vectorizer = None
bow_matrix = None
tfidf_vectorizer = None
tfidf_matrix = None
sbert_model = None
sbert_embeddings = None

@app.on_event("startup")
async def load_models():
    global df, bow_vectorizer, bow_matrix, tfidf_vectorizer, tfidf_matrix, sbert_model, sbert_embeddings
    
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH).fillna('')
    
    tfidf_features = (
        (df['job_title'] + ' ') * 3 + 
        (df['category'] + ' ') * 2 + 
        (df['skills_required'].str.replace(';', ' ') + ' ') * 3 + 
        df['job_description']
    )
    
    semantic_text = (
        "Job Title: " + df['job_title'] + ". " +
        "Category: " + df['category'] + ". " +
        "Skills: " + df['skills_required'].str.replace(';', ', ') + ". " +
        "Description: " + df['job_description']
    )
    
    print("Initializing BoW...")
    bow_vectorizer = CountVectorizer(stop_words='english')
    bow_matrix = bow_vectorizer.fit_transform(tfidf_features)

    print("Initializing TF-IDF...")
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(tfidf_features)
    
    print(f"Loading SBERT model ({MODEL_NAME}) on {DEVICE}...")
    sbert_model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    sbert_embeddings = sbert_model.encode(semantic_text.tolist(), show_progress_bar=True)
    print("Startup complete.")

# --- API Models ---
class RecommendationRequest(BaseModel):
    query: str
    user_location: str = "Phnom Penh"
    top_n: int = 5
    model_type: str = "sbert"

class ReturneeRequest(BaseModel):
    query: str = ""
    user_location: str = "Phnom Penh"
    top_n: int = 10
    model_type: str = "sbert"

class JobRecommendation(BaseModel):
    job_title: str
    company_name: str
    job_location: str
    job_type: str = "Unknown"
    score: float
    experience: str = ""
    education: str = ""
    skills: list[str] = []

# --- Routes ---
@app.get("/")
def read_root():
    return {"message": "littlehelp API is running"}

@app.post("/recommend", response_model=list[JobRecommendation])
async def recommend(request: RecommendationRequest):
    if df is None:
        raise HTTPException(status_code=503, detail="Models are still loading")
    
    m_type = request.model_type.lower()
    
    # 1. BoW Score
    q_bow = bow_vectorizer.transform([request.query])
    sims_bow = cosine_similarity(q_bow, bow_matrix).flatten()
    
    # 2. TF-IDF Score
    q_tfidf = tfidf_vectorizer.transform([request.query])
    sims_tfidf = cosine_similarity(q_tfidf, tfidf_matrix).flatten()
    
    # 3. SBERT Score
    instruction = "Represent this sentence for searching relevant passages: "
    q_sbert = sbert_model.encode([instruction + request.query], show_progress_bar=False)
    sims_sbert = cosine_similarity(q_sbert, sbert_embeddings).flatten()
    
    # 4. Decide Base Score
    if m_type == "bow":
        base_scores = sims_bow
    elif m_type == "sbert":
        base_scores = sims_sbert
    else: # Default to TF-IDF
        base_scores = sims_tfidf
    
    # 5. Apply Geometry-Correct Penalty
    loc_weights = np.array([get_geometry_weight(request.user_location, loc) for loc in df['job_location']])
    final_scores = base_scores * loc_weights
    
    # 6. Get Top N results
    top_idx = final_scores.argsort()[-request.top_n:][::-1]
    
    recommendations = []
    for idx in top_idx:
        job = df.iloc[idx]
        skills_list = [s.strip() for s in str(job['skills_required']).split(';') if s.strip()]
            
        recommendations.append(JobRecommendation(
            job_title=job['job_title'],
            company_name=job['company_name'],
            job_location=str(job['job_location']),
            job_type=str(job['job_type']),
            score=float(final_scores[idx]),
            experience=str(job['experience_required']),
            education=str(job['education_required']),
            skills=skills_list
        ))
    return recommendations

@app.post("/returnee", response_model=list[JobRecommendation])
async def returnee_jobs(request: ReturneeRequest):
    if df is None:
        raise HTTPException(status_code=503, detail="Models are still loading")
    
    try:
        exp_num = pd.to_numeric(df['experience_required'], errors='coerce').fillna(0)
        edu_allowed = ['none', 'high school', 'secondary school', 'primary school', 'vocation', '']
        mask = (
            (exp_num <= 1) & 
            (df['education_required'].str.lower().isin(edu_allowed))
        )
    except Exception as e:
        print(f"Masking error: {e}")
        return []

    if not request.query.strip():
        returnee_df = df[mask].copy()
        if returnee_df.empty:
            return []
            
        loc_weights = np.array([get_geometry_weight(request.user_location, loc) for loc in returnee_df['job_location']])
        returnee_df['loc_score'] = loc_weights
        top_df = returnee_df.sort_values('loc_score', ascending=False).head(request.top_n)
        
        results = []
        for _, job in top_df.iterrows():
            skills_list = [s.strip() for s in str(job['skills_required']).split(';') if s.strip()]
            results.append(JobRecommendation(
                job_title=job['job_title'],
                company_name=job['company_name'],
                job_location=str(job['job_location']),
                job_type=str(job['job_type']),
                score=float(job['loc_score']),
                experience=str(job['experience_required']),
                education=str(job['education_required']),
                skills=skills_list
            ))
        return results

    m_type = request.model_type.lower()
    
    # 1. Similarity Scores
    q_bow = bow_vectorizer.transform([request.query])
    sims_bow = cosine_similarity(q_bow, bow_matrix).flatten()
    
    q_tfidf = tfidf_vectorizer.transform([request.query])
    sims_tfidf = cosine_similarity(q_tfidf, tfidf_matrix).flatten()
    
    instruction = "Represent this sentence for searching relevant passages: "
    q_sbert = sbert_model.encode([instruction + request.query], show_progress_bar=False)
    sims_sbert = cosine_similarity(q_sbert, sbert_embeddings).flatten()
    
    # 2. Decide Base Score
    if m_type == "bow":
        base_scores = sims_bow
    elif m_type == "sbert":
        base_scores = sims_sbert
    else: # Default to TF-IDF
        base_scores = sims_tfidf
    
    # 3. Apply Geometry-Correct Penalty
    loc_weights = np.array([get_geometry_weight(request.user_location, loc) for loc in df['job_location']])
    final_scores = base_scores * loc_weights
    
    # 4. Filter by Returnee Mask
    masked_scores = np.where(mask, final_scores, -1.0)
    
    # 5. Get Top N results
    top_idx = masked_scores.argsort()[-request.top_n:][::-1]
    
    results = []
    for idx in top_idx:
        if masked_scores[idx] < 0:
            continue
            
        job = df.iloc[idx]
        skills_list = [s.strip() for s in str(job['skills_required']).split(';') if s.strip()]
            
        results.append(JobRecommendation(
            job_title=job['job_title'],
            company_name=job['company_name'],
            job_location=str(job['job_location']),
            job_type=str(job['job_type']),
            score=float(masked_scores[idx]),
            experience=str(job['experience_required']),
            education=str(job['education_required']),
            skills=skills_list
        ))
    return results

@app.get("/dashboard-stats")
async def get_dashboard_stats(location: str = "All"):
    if df is None:
        raise HTTPException(status_code=503, detail="Data is still loading")
    
    # Filter by location if not "All"
    display_df = df
    if location != "All":
        display_df = df[df['job_location'] == location]
    
    if display_df.empty:
        return {
            "total_jobs": 0,
            "jobs_by_province": [],
            "jobs_by_category": [],
            "top_skills": [],
            "avg_salary_by_category": [],
            "experience_distribution": []
        }

    # 1. Total Jobs
    total_jobs = len(display_df)
    
    # 2. Jobs by Location (Top 10)
    location_counts = display_df['job_location'].value_counts().head(10).to_dict()
    jobs_by_location = [{"name": k, "value": int(v)} for k, v in location_counts.items()]
    
    # 3. Jobs by Category (Top 10)
    category_counts = display_df['category'].value_counts().head(10).to_dict()
    jobs_by_category = [{"name": k, "value": int(v)} for k, v in category_counts.items()]
    
    # 4. Job Type Distribution
    type_counts = display_df['job_type'].value_counts().head(10).to_dict()
    job_type_distribution = [{"name": k, "value": int(v)} for k, v in type_counts.items() if k and k.lower() != 'nan']
    
    # 5. Education Required
    edu_counts = display_df['education_required'].value_counts().head(10).to_dict()
    education_distribution = [{"name": k if k else "Not Specified", "value": int(v)} for k, v in edu_counts.items()]

    # 6. Avg Salary by Industry (Category)
    temp_df = display_df.copy()
    temp_df['salary_avg'] = pd.to_numeric(temp_df['salary_avg'], errors='coerce')
    salary_df = temp_df[temp_df['salary_avg'] > 0]
    
    if not salary_df.empty:
        avg_salaries = salary_df.groupby('category')['salary_avg'].mean().sort_values(ascending=False).head(10).to_dict()
        avg_salary_by_industry = [{"name": k, "value": round(float(v), 2)} for k, v in avg_salaries.items()]
        
        # 7. Salary Distribution (Bins)
        sal_bins = [0, 300, 600, 1000, 2000, 5000, 100000]
        sal_labels = ['<$300', '$300-600', '$600-1k', '$1k-2k', '$2k-5k', '$5k+']
        sal_dist = pd.cut(salary_df['salary_avg'], bins=sal_bins, labels=sal_labels).value_counts().sort_index().to_dict()
        salary_distribution = [{"name": k, "value": int(v)} for k, v in sal_dist.items()]
    else:
        avg_salary_by_industry = []
        salary_distribution = []

    return {
        "total_jobs": total_jobs,
        "jobs_by_location": jobs_by_location,
        "jobs_by_category": jobs_by_category,
        "job_type_distribution": job_type_distribution,
        "education_distribution": education_distribution,
        "salary_distribution": salary_distribution,
        "avg_salary_by_industry": avg_salary_by_industry
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
