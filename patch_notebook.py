import json

with open('test/main.ipynb', 'r') as f:
    nb = json.load(f)

new_source = """manual_profiles = [
    {"user_id": 1, "name": "User 1", "job title": "Delivery Driver", "skill": "driving, motorbike, city knowledge", "preferenced_location": "Phnom Penh", "preferennce_category": "Logistics/Transport", "experience_years": 2},
    {"user_id": 2, "name": "User 2", "job title": "Security Guard", "skill": "security, honesty, hardworking", "preferenced_location": "Phnom Penh", "preferennce_category": "Security", "experience_years": 1},
    {"user_id": 3, "name": "User 3", "job title": "Software Developer", "skill": "Python, React, SQL", "preferenced_location": "Remote", "preferennce_category": "IT/Software Development", "experience_years": 3},
    {"user_id": 4, "name": "User 4", "job title": "Sales Assistant", "skill": "communication, customer service", "preferenced_location": "Phnom Penh", "preferennce_category": "Sales/Retail", "experience_years": 1},
    {"user_id": 5, "name": "User 5", "job title": "Waitress/Waiter", "skill": "English, food service", "preferenced_location": "Phnom Penh", "preferennce_category": "Hotel/Hospitality", "experience_years": 2},
    {"user_id": 6, "name": "User 6", "job title": "Accountant", "skill": "QuickBooks, Excel, tax, payroll", "preferenced_location": "Phnom Penh", "preferennce_category": "Accounting/Finance", "experience_years": 3},
    {"user_id": 7, "name": "User 7", "job title": "Construction Labor", "skill": "site clearing, heavy lifting", "preferenced_location": "Phnom Penh", "preferennce_category": "Construction", "experience_years": 2},
    {"user_id": 8, "name": "User 8", "job title": "Hotel Receptionist", "skill": "English, Chinese, reception", "preferenced_location": "Phnom Penh", "preferennce_category": "Hotel/Hospitality", "experience_years": 1},
    {"user_id": 9, "name": "User 9", "job title": "Cleaner", "skill": "cleaning, punctual, tidy", "preferenced_location": "Phnom Penh", "preferennce_category": "Cleaning/Janitorial", "experience_years": 2},
    {"user_id": 10, "name": "User 10", "job title": "Admin Assistant", "skill": "typing, filing, office management", "preferenced_location": "Phnom Penh", "preferennce_category": "Admin/Human Resources", "experience_years": 2}
]

def run_comprehensive_manual_study():
    global manual_results, manual_profile_metrics
    manual_results = {'bow': [], 'tfidf': [], 'sbert': []}
    instruction = "Represent this sentence for searching relevant passages: "
    
    category_mapping = {
        "Logistics/Transport": "logistics",
        "Security": "security",
        "IT/Software Development": "technology",
        "Sales/Retail": "retail",
        "Hotel/Hospitality": "hospitality",
        "Accounting/Finance": "accounting",
        "Construction": "construction",
        "Cleaning/Janitorial": "cleaning",
        "Admin/Human Resources": "human resources"
    }

    print(f"{'PROFILE':<20} | {'MODEL':<8} | {'TOP 3 RECOMMENDATIONS (Score - Category)'}")
    print("-" * 110)

    for p in manual_profiles:
        query = f"I am a {p['job title']} in {p['preferenced_location']}. My skills are {p['skill']}. I have {p['experience_years']} years of experience."
        target_cat_full = p['preferennce_category']
        target_cat_sub = category_mapping.get(target_cat_full, target_cat_full.lower())
        
        relevant_mask = df_train['category'].str.lower().str.contains(target_cat_sub).values
        total_relevant = sum(relevant_mask)
        
        q_bow = bow_vectorizer.transform([query])
        s_bow = cosine_similarity(q_bow, bow_matrix).flatten()
        
        q_tfidf = tfidf_vectorizer.transform([query])
        s_tfidf = cosine_similarity(q_tfidf, tfidf_matrix).flatten()
        
        q_sbert = sbert_model.encode([instruction + query], show_progress_bar=False)
        s_sbert = cosine_similarity(q_sbert.reshape(1, -1), sbert_embeddings).flatten()
        
        models = [('bow', s_bow), ('tfidf', s_tfidf), ('sbert', s_sbert)]
        
        print(f"{p['job title']:<20} (Expected: {target_cat_full})")
        
        for m_key, scores in models:
            sorted_indices = scores.argsort()[::-1]
            top_idx = sorted_indices[:10]
            
            recs = []
            for i in range(3):
                job = df_train.iloc[sorted_indices[i]]
                recs.append(f"{job['job_title']} ({scores[sorted_indices[i]]:.2f} - {job['category']})")
            
            print(f"{'':<20} | {m_key.upper():<8} | 1. {recs[0]}")
            print(f"{'':<20} | {'':<8} | 2. {recs[1]}")
            print(f"{'':<20} | {'':<8} | 3. {recs[2]}")
            
            if total_relevant > 0:
                is_relevant_top_10 = relevant_mask[top_idx]
                p10 = sum(is_relevant_top_10) / 10
                r10 = sum(is_relevant_top_10) / total_relevant
                
                is_relevant_top_100 = relevant_mask[sorted_indices[:100]]
                map_val = calculate_map_cat(is_relevant_top_100, total_relevant)
                ndcg10 = calculate_ndcg_cat(is_relevant_top_10, total_relevant, k=10)
                
                manual_results[m_key].append([p10, r10, map_val, ndcg10])
                
        print("-" * 110)
    
    # --- Visualization ---
    manual_profile_metrics = []
    for model in ['bow', 'tfidf', 'sbert']:
        m_data = manual_results[model]
        if not m_data: continue
        
        m = np.mean(m_data, axis=0)
        manual_profile_metrics.append({'Model': model.upper(), 'Metric': 'Precision @10', 'Score (%)': m[0] * 100})
        manual_profile_metrics.append({'Model': model.upper(), 'Metric': 'Recall @10', 'Score (%)': m[1] * 100})
        manual_profile_metrics.append({'Model': model.upper(), 'Metric': 'MAP @100', 'Score (%)': m[2] * 100})
        manual_profile_metrics.append({'Model': model.upper(), 'Metric': 'NDCG @10', 'Score (%)': m[3] * 100})

    if manual_profile_metrics:
        plt.figure(figsize=(12, 7))
        sns.barplot(data=pd.DataFrame(manual_profile_metrics), x='Metric', y='Score (%)', hue='Model', palette={'BOW': 'blue', 'TFIDF': 'green', 'SBERT': 'orange'})
        plt.title('Manual Profile Recommendation Performance (n=10)', fontsize=15, fontweight='bold')
        plt.ylim(0, 110)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.show()

run_comprehensive_manual_study()
"""

# Format as lines for ipynb
new_source_lines = [line + '\n' for line in new_source.split('\n')]
# Remove the trailing newline on the last line
if new_source_lines:
    new_source_lines[-1] = new_source_lines[-1].rstrip('\n')

nb['cells'][-1]['source'] = new_source_lines

with open('test/main.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
