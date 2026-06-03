import pandas as pd
import numpy as np
import faiss
import joblib
from sentence_transformers import SentenceTransformer
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'saved_model')

class JobRecommender:
    def __init__(self):
        print("Loading recommendation model...")
        # 1. Load Metadata
        self.metadata = pd.read_parquet(os.path.join(MODEL_DIR, 'metadata.parquet'))
        
        # 2. Load FAISS Index
        self.index = faiss.read_index(os.path.join(MODEL_DIR, 'sbert_index.faiss'))
        
        # 3. Load SBERT Model (for encoding the user query)
        # Using CPU for inference is usually fine and saves VRAM
        self.model = SentenceTransformer('BAAI/bge-base-en-v1.5', device='cpu')
        
        # 4. Load TF-IDF (Optional, for keyword boosting)
        self.tfidf_vec = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib'))
        
        print(f"Model loaded with {len(self.metadata)} jobs.")

    def get_recommendations(self, query, top_n=5):
        # 1. Encode the query
        instruction = "Represent this sentence for searching relevant passages: "
        query_embedding = self.model.encode([instruction + query], convert_to_numpy=True)
        
        # 2. Normalize for Cosine Similarity
        faiss.normalize_L2(query_embedding)
        
        # 3. Search Index
        distances, indices = self.index.search(query_embedding, top_n)
        
        # 4. Format Results
        results = []
        for i in range(top_n):
            idx = indices[0][i]
            score = distances[0][i]
            job_info = self.metadata.iloc[idx].to_dict()
            
            # Standardize output for the frontend
            # The frontend expects these keys based on App.py's JobRecommendation model
            # and previous interaction history.
            job_info['match_score'] = float(score)
            
            # Ensure description is string
            if 'job_description' in job_info:
                job_info['job_description'] = str(job_info['job_description'])
                
            results.append(job_info)
            
        return results
            
    def get_all_jobs(self):
        return self.metadata

# Example Usage
if __name__ == "__main__":
    recommender = JobRecommender()
    
    my_query = "I am looking for a delivery driver job or warehouse work"
    print(f"\nQuery: {my_query}")
    
    matches = recommender.get_recommendations(my_query)
    
    for match in matches:
        print(f"- {match['Job Title']} at {match['Company']} ({match['location']}) - {match['match_score']*100:.1f}% Match")
