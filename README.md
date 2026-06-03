# littlehelp

## Mini Project Assignment

## Run this program

Download project to your local machine.

```bash
git clone https://github.com/Radayou07/Job_Recommendation_system.git

# use frontend directory
cd frontend
npm install # make sure nodejs is installed in your system
npm run dev # run the server

# open backend directory
cd backend
pip install -r requirements.txt
python App.py # run App.py
```

>[!note]
For Nvidia product with CUDA support install this instead for faster train data.

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124
```

since with CUDA it will run faster than run on CPU.

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
* **Model Used:** `BAAI/bge-base-en-v1.5`
* **Purpose:** Our most advanced model. Instead of just matching literal keywords, SBERT captures the actual contextual meaning and intent behind candidate profiles and job descriptions, delivering highly accurate, context-aware recommendations.

### 4. Hybrid Model (The "Brains")

* **Type:** Multi-Strategy Ensemble (SBERT + TF-IDF)
* **Purpose:** Combines the keyword precision of TF-IDF with the semantic understanding of SBERT. By using a weighted approach (70% SBERT, 30% TF-IDF), the system delivers matches that are both technically accurate and contextually relevant.
