import pandas as pd
import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os

# Configuration - using absolute paths to be safe
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '../backend/data/data.csv')
MODEL_NAME = "BAAI/bge-base-en-v1.5"
DEVICE = 'cpu'

# 1. Define 10 User Profiles with Expected Categories
profiles = [
    {"name": "Delivery Driver", "query": "I am looking for a delivery driver job in Phnom Penh. I have a motorbike and know the city well.", "expected": "Logistics/Transport"},
    {"name": "Security Guard", "query": "Security guard, honest and hardworking, 1 year experience, looking for night shift.", "expected": "Security"},
    {"name": "Software Developer", "query": "Full stack developer skilled in Python, React, and SQL. 3 years of experience.", "expected": "IT/Software Development"},
    {"name": "Sales Assistant", "query": "Sales assistant for a retail shop or supermarket. Good communication and customer service.", "expected": "Sales/Retail"},
    {"name": "Waitress/Waiter", "query": "Experience in international restaurants, good English, looking for server position.", "expected": "Hotel/Hospitality"},
    {"name": "Accountant", "query": "Accountant with knowledge of QuickBooks and Excel. I can manage tax and payroll.", "expected": "Accounting/Finance"},
    {"name": "Construction Labor", "query": "Strong construction worker, experienced in site clearing and heavy lifting.", "expected": "Construction"},
    {"name": "Hotel Receptionist", "query": "Receptionist for hotel or guesthouse. Fluent in English and Chinese.", "expected": "Hotel/Hospitality"},
    {"name": "Cleaner", "query": "Office cleaner or maid. Diligent, punctual, and very tidy.", "expected": "Cleaning/Janitorial"},
    {"name": "Admin Assistant", "query": "Administrative assistant, good at typing, filing documents, and office management.", "expected": "Admin/Human Resources"}
]

def run_test():
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    # 2. Load and Preprocess Data
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

    # 3. Initialize Models
    print("Initializing BoW...")
    bow_vec = CountVectorizer(stop_words='english')
    bow_matrix = bow_vec.fit_transform(tfidf_features)

    print("Initializing TF-IDF...")
    tfidf_vec = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vec.fit_transform(tfidf_features)
    
    print(f"Loading SBERT model ({MODEL_NAME})...")
    sbert_model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    sbert_embeddings = sbert_model.encode(semantic_text.tolist(), show_progress_bar=False)

    # 4. Test each profile
    print("\n" + "="*110)
    print(f"{'PROFILE':<20} | {'MODEL':<8} | {'TOP 3 RECOMMENDATIONS (Score - Category)'}")
    print("-" * 110)

    instruction = "Represent this sentence for searching relevant passages: "
    
    hits = {'BoW': 0, 'TF-IDF': 0, 'SBERT': 0}

    for p in profiles:
        query = p['query']
        expected = p['expected'].lower()
        
        # Calculate similarities
        q_bow = bow_vec.transform([query])
        sim_bow = cosine_similarity(q_bow, bow_matrix).flatten()
        
        q_tfidf = tfidf_vec.transform([query])
        sim_tfidf = cosine_similarity(q_tfidf, tfidf_matrix).flatten()
        
        q_sbert = sbert_model.encode([instruction + query], show_progress_bar=False)
        sim_sbert = cosine_similarity(q_sbert, sbert_embeddings).flatten()
        
        models = [
            ('BoW', sim_bow),
            ('TF-IDF', sim_tfidf),
            ('SBERT', sim_sbert)
        ]

        print(f"{p['name']:<20} (Expected: {p['expected']})")
        
        for m_name, scores in models:
            top_indices = scores.argsort()[-3:][::-1]
            recs = []
            for i, idx in enumerate(top_indices):
                job = df.iloc[idx]
                cat = job['category']
                title = job['job_title']
                score = scores[idx]
                
                # Check for category hit on Top #1 result
                if i == 0 and expected in cat.lower():
                    hits[m_name] += 1
                
                recs.append(f"{title} ({score:.2f} - {cat})")
            
            print(f"{'':<20} | {m_name:<8} | 1. {recs[0]}")
            print(f"{'':<20} | {'':<8} | 2. {recs[1]}")
            print(f"{'':<20} | {'':<8} | 3. {recs[2]}")
        print("-" * 110)

    # 5. Final Summary
    print("\nSUMMARY: Category Match Accuracy (Top 1 Recommendation)")
    for m, count in hits.items():
        print(f"{m:<8}: {count}/{len(profiles)} ({count/len(profiles)*100:.1f}%)")

if __name__ == "__main__":
    run_test()
