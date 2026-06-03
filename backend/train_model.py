import pandas as pd
import numpy as np
import faiss
import joblib
from sentence_transformers import SentenceTransformer
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'data.csv')
SAVE_DIR = os.path.join(BASE_DIR, 'saved_model')

def train_model():
    print("Loading data for training...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"Training on {len(df)} jobs.")

    # 1. Initialize SBERT
    # Using a high-quality model for best accuracy
    model = SentenceTransformer('BAAI/bge-base-en-v1.5')
    
    # 2. Prepare text for embedding
    # Combine title, category, and description for rich semantic search
    df['combined_text'] = (
        "Job Title: " + df['job_title'].fillna('') + ". " +
        "Category: " + df['category'].fillna('') + ". " +
        "Description: " + df['job_description'].fillna('') + ". " +
        "Skills: " + df['skills_required'].fillna('')
    )

    print("Generating embeddings (this may take a few minutes on CPU)...")
    # Add instruction for BGE model
    instruction = "Represent this sentence for searching relevant passages: "
    texts = [instruction + t for t in df['combined_text'].tolist()]
    
    # Encode
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    # 3. Normalize for Cosine Similarity in FAISS
    faiss.normalize_L2(embeddings)
    
    # 4. Build FAISS Index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension) # Inner Product on normalized vectors = Cosine Similarity
    index.add(embeddings)
    
    # 5. Save Artifacts
    print(f"Saving artifacts to {SAVE_DIR}...")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Save Index
    faiss.write_index(index, os.path.join(SAVE_DIR, 'sbert_index.faiss'))
    
    # Save Metadata (Parquet is faster and smaller than CSV)
    df.drop(columns=['combined_text']).to_parquet(os.path.join(SAVE_DIR, 'metadata.parquet'), index=False)
    
    # Save simple TF-IDF for keyword matching fallback if needed
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf.fit(df['combined_text'])
    joblib.dump(tfidf, os.path.join(SAVE_DIR, 'tfidf_vectorizer.joblib'))
    
    print("Training complete!")

if __name__ == "__main__":
    train_model()
