import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
import joblib 
import requests

# -----------------------------
# 🔹 Create embedding
# -----------------------------
def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    if r.status_code != 200:
        raise Exception("❌ Embedding API error")

    return r.json()["embeddings"]

# -----------------------------
# 🔹 LLM response
# -----------------------------
def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    })

    if r.status_code != 200:
        raise Exception("❌ LLM API error")

    return r.json()["response"]

# -----------------------------
# 🔹 Load data
# -----------------------------
df = joblib.load('embeddings.joblib')

# -----------------------------
# 🔹 Show topics
# -----------------------------
print("\n📚 Topics Covered:")
topics = df['topic'].unique()

for t in topics:
    print("-", t)

# -----------------------------
# 🔹 Show suggestions
# -----------------------------
print("\n💡 Try asking:")

for i, row in df.head(5).iterrows():
    print("-", row['preview'])

# -----------------------------
# 🔹 Take user query
# -----------------------------
incoming_query = input("\n❓ Ask a Question: ")

# -----------------------------
# 🔹 Create query embedding
# -----------------------------
question_embedding = create_embedding([incoming_query])[0] 

# -----------------------------
# 🔹 Similarity search
# -----------------------------
similarities = cosine_similarity(
    np.vstack(df['embedding']), 
    [question_embedding]
).flatten()

# -----------------------------
# 🔹 Check relevance
# -----------------------------
if max(similarities) < 0.3:
    print("\n❌ Question not related to video content")
    exit()

# -----------------------------
# 🔹 Get top results
# -----------------------------
top_results = 5
max_indx = similarities.argsort()[::-1][:top_results]
new_df = df.loc[max_indx]

# -----------------------------
# 🔹 Show best timestamp (quick answer)
# -----------------------------
best_row = new_df.iloc[0]

minutes = int(best_row['start'] // 60)
seconds = int(best_row['start'] % 60)

print("\n🎯 Best Match:")
print(f"⏱️ Go to: {minutes:02d}:{seconds:02d}")
print("📌", best_row['text'])

# -----------------------------
# 🔹 Build LLM prompt
# -----------------------------
prompt = f"""
You are an AI teaching assistant.

Here are video chunks with:
- title
- video number
- start time
- end time
- text

{new_df[['title', 'number', 'start', 'end', 'text']].to_json(orient='records')}

---------------------------------

User Question:
{incoming_query}

Instructions:
- Answer clearly in simple human language
- Mention video title and timestamp
- Guide user where to go
- If unrelated, say you only answer from video
"""

# -----------------------------
# 🔹 Get AI answer
# -----------------------------
response = inference(prompt)

print("\n🤖 Detailed Answer:")
print(response)

# -----------------------------
# 🔹 Save logs (optional)
# -----------------------------
with open("prompt.txt", "w", encoding="utf-8") as f:
    f.write(prompt)

with open("response.txt", "w", encoding="utf-8") as f:
    f.write(response)