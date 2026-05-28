# Job Recommendation

## Mini Project Assignment

## Objective

so the goal is to build a recommendation model that help ## Returnee ## to find a job easier. 

## Models Implemented

### 1. Bag of Words (BoW)

* **Type:** Frequency-based Vectorization
* **Purpose:** Serves as our foundational baseline. While simple, implementing BoW is the critical first step to understanding text preprocessing, tokenization, and basic NLP pipelines.

### 2. Term Frequency-Inverse Document Frequency (TF-IDF)

* **Type:** Weighted Keyword Vectorization
* **Purpose:** Steps up the complexity by accounting for word importance. TF-IDF down-weights universally common words (like "the" or "and") and highlights unique keywords that define specific job roles and resumes.

### 3. Sentence-BERT (SBERT)

* **Type:** Deep Learning Semantic Embeddings
* **Model Used:** `all-MiniLM-L6-v2`
* **Purpose:** Our most advanced model. Instead of just matching literal keywords, SBERT captures the actual contextual meaning and intent behind candidate profiles and job descriptions, delivering highly accurate, context-aware recommendations.
