import pandas as pd
import numpy as np
import torch
import math
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
import os

# 1. Load Data
# Use absolute path to avoid directory issues
DATA_PATH = '/media/radayou/DATA/Study/ITC/Year 3/2nd semester/Mini Project/Final/Job_Recommendation/backend/data/data.csv'
df = pd.read_csv(DATA_PATH).fillna('')

# Pre-processing
df['tfidf_features'] = (df['job_title'] + ' ') * 3 + (df['category'] + ' ') * 2 + (df['skills_required'].str.replace(';', ' ') + ' ') * 3 + df['job_description']
df['semantic_text'] = "Job Title: " + df['job_title'] + ". Category: " + df['category'] + ". Skills: " + df['skills_required'].str.replace(';', ', ')

# 2. Split (95/5)
df_train, df_val = train_test_split(df, test_size=0.05, random_state=42)

# 3. Initialize Models
bow_vec = CountVectorizer(stop_words='english').fit(df_train['tfidf_features'])
bow_matrix = bow_vec.transform(df_train['tfidf_features'])

tfidf_vec = TfidfVectorizer(stop_words='english').fit(df_train['tfidf_features'])
tfidf_matrix = tfidf_vec.transform(df_train['tfidf_features'])

sbert_model = SentenceTransformer('BAAI/bge-base-en-v1.5', device='cpu')
train_embeddings = sbert_model.encode(df_train['semantic_text'].tolist(), show_progress_bar=False)

# 4. Evaluation Functions
def calculate_map(is_relevant):
    if sum(is_relevant) == 0: return 0
    p_sum = 0
    hits = 0
    for i, rel in enumerate(is_relevant):
        if rel:
            hits += 1
            p_sum += hits / (i + 1)
    return p_sum / sum(is_relevant)

def calculate_ndcg(is_relevant_top_k, total_relevant_count, k=10):
    dcg = 0
    for i in range(min(len(is_relevant_top_k), k)):
        if is_relevant_top_k[i]:
            dcg += 1 / math.log2(i + 2)
    
    # Ideal DCG: top relevant items are at the top
    idcg = sum([1 / math.log2(i + 2) for i in range(min(total_relevant_count, k))])
    return dcg / idcg if idcg > 0 else 0

# 5. Run Evaluation
results = {'bow': [], 'tfidf': [], 'sbert': []}

# We treat the validation jobs as "User Profiles"
# Query = Expected Title + Skills
val_queries = "Job Title: " + df_val['job_title'] + ". Skills: " + df_val['skills_required']

print(f"Evaluating {len(df_val)} User Profiles...")

for i in range(len(df_val)):
    profile = df_val.iloc[i]
    query_text = val_queries.iloc[i]
    target_category = profile['category']
    
    # Define Ground Truth: All jobs in the training set with the same category
    relevant_mask = (df_train['category'] == target_category).values
    total_relevant = sum(relevant_mask)
    if total_relevant == 0: continue

    # Get Scores
    q_bow = bow_vec.transform([query_text])
    q_tfidf = tfidf_vec.transform([query_text])
    q_sbert = sbert_model.encode(["Represent this sentence for searching relevant passages: " + query_text], show_progress_bar=False)

    scores_dict = {
        'bow': cosine_similarity(q_bow, bow_matrix).flatten(),
        'tfidf': cosine_similarity(q_tfidf, tfidf_matrix).flatten(),
        'sbert': cosine_similarity(q_sbert, train_embeddings).flatten()
    }

    for model, scores in scores_dict.items():
        sorted_indices = scores.argsort()[::-1]
        top_idx = sorted_indices[:10]
        is_relevant_top_10 = relevant_mask[top_idx]
        
        p10 = sum(is_relevant_top_10) / 10
        r10 = sum(is_relevant_top_10) / total_relevant
        
        # MAP usually requires full ranking, but we'll cap it at 100 for speed
        map_val = calculate_map(relevant_mask[sorted_indices[:100]]) 
        ndcg10 = calculate_ndcg(is_relevant_top_10, total_relevant, k=10)
        
        results[model].append([p10, r10, map_val, ndcg10])

# 6. Aggregate
print("\nResults (User Profile Recommendation):")
print(f"{'Model':<10} | {'P@10':<8} | {'R@10':<8} | {'MAP@100':<8} | {'NDCG@10':<8}")
print("-" * 65)
for model in ['bow', 'tfidf', 'sbert']:
    m = np.mean(results[model], axis=0)
    print(f"{model.upper():<10} | {m[0]:.4f}   | {m[1]:.4f}   | {m[2]:.4f}   | {m[3]:.4f}")
