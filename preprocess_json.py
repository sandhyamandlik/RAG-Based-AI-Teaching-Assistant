import requests
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import re

# -----------------------------
# 🔹 Clean text function
# -----------------------------
def clean_text(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)  # remove extra spaces
    return text

# -----------------------------
# 🔹 Create embeddings
# -----------------------------
def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    if r.status_code != 200:
        raise Exception("❌ Embedding API failed")

    return r.json()["embeddings"]

# -----------------------------
# 🔹 Process JSON files
# -----------------------------
jsons = os.listdir("jsons")

my_dicts = []
chunk_id = 0

for json_file in jsons:
    with open(f"jsons/{json_file}", encoding="utf-8") as f:
        content = json.load(f)

    print(f"🔄 Processing: {json_file}")

    # 🔹 Clean all texts first
    texts = []
    valid_chunks = []

    for chunk in content['chunks']:
        text = clean_text(chunk['text'])

        # ❌ Skip useless chunks
        if len(text) < 10:
            continue

        texts.append(text)
        valid_chunks.append(chunk)

    if len(texts) == 0:
        continue

    # 🔹 Create embeddings
    embeddings = create_embedding(texts)

    # 🔹 Store structured data
    for i, chunk in enumerate(valid_chunks):
        cleaned_text = clean_text(chunk['text'])

        chunk_data = {
            "chunk_id": chunk_id,
            "number": chunk.get("number"),
            "title": chunk.get("title"),
            "start": chunk.get("start"),
            "end": chunk.get("end"),
            "text": cleaned_text,
            "embedding": embeddings[i],

            # 🔥 NEW FIELDS (SMART FEATURES)
            "preview": cleaned_text[:80],  # short preview
            "topic": chunk.get("title", "").replace("-", " ")
        }

        my_dicts.append(chunk_data)
        chunk_id += 1

# -----------------------------
# 🔹 Create DataFrame
# -----------------------------
df = pd.DataFrame.from_records(my_dicts)

print("\n✅ Total chunks processed:", len(df))

# -----------------------------
# 🔹 Save embeddings
# -----------------------------
joblib.dump(df, "embeddings.joblib")

print("💾 Saved to embeddings.joblib")