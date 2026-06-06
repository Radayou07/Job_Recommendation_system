
import torch
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
import math
import os

# 1. Data Preparation
DATA_PATH = '../backend/data/data.csv'
if not os.path.exists(DATA_PATH):
    # Try absolute path or different relative path if needed, but the notebook uses this
    DATA_PATH = 'Final/Job_Recommendation/backend/data/data.csv'

df = pd.read_csv(DATA_PATH).fillna('')

# Pre-processing matching App.py
df['tfidf_features'] = (
    (df['job_title'] + ' ') * 3 + 
    (df['category'] + ' ') * 2 + 
    (df['skills_required'].str.replace(';', ' ') + ' ') * 3 + 
    df['job_description']
)

df['semantic_text'] = (
    "Job Title: " + df['job_title'] + ". " +
    "Category: " + df['category'] + ". " +
    "Skills: " + df['skills_required'].str.replace(';', ', ') + ". " +
    "Description: " + df['job_description']
)

# 2. Split
df_train, df_val = train_test_split(df, test_size=0.05, random_state=42)

# 3. Models
print("Initializing Models...")
bow_vectorizer = CountVectorizer(stop_words='english')
bow_matrix = bow_vectorizer.fit_transform(df_train['tfidf_features'])

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df_train['tfidf_features'])

device = 'cpu'
sbert_model = SentenceTransformer('BAAI/bge-base-en-v1.5', device=device)
sbert_embeddings = sbert_model.encode(df_train['semantic_text'].tolist(), show_progress_bar=True)

# 4. Evaluation Helpers
def calculate_map_cat(is_relevant_list, total_rel):
    p_sum, hits = 0, 0
    for i, rel in enumerate(is_relevant_list):
        if rel:
            hits += 1
            p_sum += hits / (i + 1)
    return p_sum / total_rel if total_rel > 0 else 0

def calculate_ndcg_cat(is_relevant_list, total_rel, k=10):
    dcg = 0
    for i in range(min(len(is_relevant_list), k)):
        if is_relevant_list[i]:
            dcg += 1 / math.log2(i + 2)
    
    idcg = sum([1 / math.log2(i + 2) for i in range(min(total_rel, k))])
    return dcg / idcg if idcg > 0 else 0

# 5. User Profile Evaluation
def evaluate_user_profiles():
    results = {'BOW': [], 'TFIDF': [], 'SBERT': []}
    val_queries = "Job Title: " + df_val['job_title'] + ". Skills: " + df_val['skills_required']
    
    print(f"Evaluating {len(df_val)} User Profiles...")
    
    # Instruction for BGE
    instruction = "Represent this sentence for searching relevant passages: "
    q_sbert_encoded = sbert_model.encode([instruction + q for q in val_queries], show_progress_bar=True)

    for i in range(len(df_val)):
        profile = df_val.iloc[i]
        query_text = val_queries.iloc[i]
        target_category = profile['category']
        
        relevant_mask = (df_train['category'] == target_category).values
        total_relevant = sum(relevant_mask)
        if total_relevant == 0: continue

        # Vectorize single query
        q_bow = bow_vectorizer.transform([query_text])
        q_tfidf = tfidf_vectorizer.transform([query_text])
        q_sbert = q_sbert_encoded[i].reshape(1, -1)

        scores_dict = {
            'BOW': cosine_similarity(q_bow, bow_matrix).flatten(),
            'TFIDF': cosine_similarity(q_tfidf, tfidf_matrix).flatten(),
            'SBERT': q_sbert @ sbert_embeddings.T # Cosine similarity for normalized embeddings
        }

        # Handle BOW and TFIDF cosine similarity
        scores_dict['BOW'] = scores_dict['BOW']
        scores_dict['TFIDF'] = scores_dict['TFIDF']
        # SBERT embeddings from sentence_transformers are usually normalized or cosine_similarity should be used
        # Let's use cosine_similarity for all for safety
        scores_dict['SBERT'] = cosine_similarity(q_sbert, sbert_embeddings).flatten()

        for model, scores in scores_dict.items():
            sorted_indices = scores.argsort()[::-1]
            
            # Top 10 relevance
            is_relevant_top_10 = relevant_mask[sorted_indices[:10]]
            
            p10 = sum(is_relevant_top_10) / 10
            r10 = sum(is_relevant_top_10) / total_relevant
            
            # For MAP and NDCG, we look at top 100 for better ranking insight
            is_relevant_top_100 = relevant_mask[sorted_indices[:100]]
            map_val = calculate_map_cat(is_relevant_top_100, total_relevant)
            ndcg10 = calculate_ndcg_cat(is_relevant_top_10, total_relevant, k=10)
            
            results[model].append([p10, r10, map_val, ndcg10])
            
    final_results = []
    for model in ['BOW', 'TFIDF', 'SBERT']:
        res = np.array(results[model])
        final_results.append({
            'Model': model,
            'Precision@10 (%)': np.mean(res[:, 0]) * 100,
            'Recall@10 (%)': np.mean(res[:, 1]) * 100,
            'MAP (%)': np.mean(res[:, 2]) * 100,
            'NDCG@10 (%)': np.mean(res[:, 3]) * 100
        })
    
    return pd.DataFrame(final_results)

results_df = evaluate_user_profiles()
print("\nFinal User Profile Evaluation Results:")
print(results_df.to_string(index=False))
