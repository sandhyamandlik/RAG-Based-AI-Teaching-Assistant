from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import requests

app = FastAPI()

# ---------------- LOAD EMBEDDINGS ----------------
try:
    df = joblib.load("embeddings.joblib")
    EMBEDDINGS_READY = True
except:
    df = pd.DataFrame()
    EMBEDDINGS_READY = False

# ---------------- CHECK OLLAMA ----------------
def check_ollama():
    try:
        requests.get("http://localhost:11434", timeout=2)
        return True
    except:
        return False

# ---------------- EMBEDDING ----------------
def create_embedding(text):
    try:
        r = requests.post(
            "http://localhost:11434/api/embed",
            json={
                "model": "bge-m3",
                "input": [text]
            },
            timeout=10
        )
        return r.json()["embeddings"][0]
    except Exception as e:
        raise Exception(f"Embedding error: {str(e)}")

# ---------------- LLM RESPONSE ----------------
def inference(prompt):
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7
                }
            },
            timeout=60
        )
        return r.json()["response"]
    except Exception as e:
        raise Exception(f"Inference error: {str(e)}")

# ---------------- STATUS API ----------------
@app.get("/api/status")
def status():
    return {
        "embeddings": {
            "count": len(df) if EMBEDDINGS_READY else 0,
            "status": "ready" if EMBEDDINGS_READY else "missing embeddings.joblib"
        },
        "ollama": {
            "running": check_ollama(),
            "model": "llama3.2"
        }
    }

# ---------------- REQUEST MODEL ----------------
class Query(BaseModel):
    query: str

# ---------------- MAIN QUERY API ----------------
@app.post("/api/query")
def query_endpoint(q: Query):

    try:
        if not EMBEDDINGS_READY or len(df) == 0:
            return {"success": False, "error": "Embeddings not loaded"}

        if not check_ollama():
            return {"success": False, "error": "Ollama not running"}

        # 🔹 Create embedding
        question_embedding = create_embedding(q.query)

        # 🔹 Similarity search
        matrix = np.vstack(df["embedding"].values)
        similarities = cosine_similarity(matrix, [question_embedding]).flatten()

        top_k = 5
        min_sim = 0.2

        idx = similarities.argsort()[::-1][:top_k]
        results = df.iloc[idx].copy()
        results["similarity"] = similarities[idx]

        results = results[results["similarity"] >= min_sim]

        if results.empty:
            return {"success": False, "error": "No relevant content found"}

        # 🔹 Context
        context = results[["title", "start", "end", "text"]].to_json(orient="records")

        # 🔥 IMPROVED PROMPT (VERY IMPORTANT)
        prompt = f"""
        You are an AI teaching assistant.

        Your job:
        - Explain the answer in detail (minimum 8-10 lines)
        - Use simple language
        - Give at least 1 real-world example
        - Guide user to exact video + timestamp

        Context:
        {context}

        Question:
        {q.query}

        Answer format:
        1. Explanation
        2. Example
        3. Where to learn (video + timestamp)
        """

        response = inference(prompt)

        return {
            "success": True,
            "response": response,
            "videos": results.to_dict(orient="records")
        }

    except Exception as e:
        return {"success": False, "error": str(e)}