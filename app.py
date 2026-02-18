from fastapi import FastAPI
from pydantic import BaseModel
from pymongo import MongoClient
import faiss
import os
from dotenv import load_dotenv
import numpy as np
import requests

# ================= INIT =================
app = FastAPI()
load_dotenv()

# ================= ENV VARIABLES =================
MONGO_URI = os.getenv("MONGO_URI")
HF_TOKEN = os.getenv("HF_TOKEN")

# ================= MONGODB =================
client = MongoClient(MONGO_URI)

try:
    client.admin.command("ping")
    print("✅ MongoDB connected successfully (FastAPI)")
except Exception as e:
    print("❌ MongoDB connection failed:", e)

db = client["test"]
signups = db["signups"]

# ================= HUGGING FACE API =================
API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-mpnet-base-v2/pipeline/feature-extraction"


HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
}

def get_embedding(texts):
    response = requests.post(
        API_URL,
        headers=HEADERS,
        json={
            "inputs": texts
        },
    )

    if response.status_code != 200:
        raise Exception(f"HuggingFace API Error: {response.text}")

    embeddings = response.json()

    return np.array(embeddings).astype("float32")


# ================= REQUEST MODEL =================
class SuggestRequest(BaseModel):
    studentSkills: str
    studentAoi: str


# ================= API ENDPOINT =================
@app.post("/suggest")
def suggest_alumni(req: SuggestRequest):

    print("📥 Incoming request:", req)

    # Fetch alumni only
    alumni_list = list(signups.find({"role": "alumni"}))

    print("📊 Total alumni found:", len(alumni_list))

    if not alumni_list:
        return []

    # ================= PREPARE TEXT =================
    alumni_texts = [
        f"{a.get('skills','')} {a.get('areaOfInterest','')} {a.get('about','')}"
        for a in alumni_list
    ]

    # ================= GET EMBEDDINGS FROM HF =================
    alumni_embeddings = get_embedding(alumni_texts)

    # Normalize vectors for cosine similarity
    faiss.normalize_L2(alumni_embeddings)

    query_text = f"{req.studentSkills} {req.studentAoi}"
    query_vec = get_embedding([query_text])

    faiss.normalize_L2(query_vec)

    # ================= FAISS INDEX =================
    dimension = alumni_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
    index.add(alumni_embeddings)

    D, I = index.search(query_vec, len(alumni_list))

    # ================= FILTER RESULTS =================
    threshold = 0.15
    results = []

    for idx, score in zip(I[0], D[0]):
        if score < threshold:
            continue

        alumni = alumni_list[idx]

        results.append({
            "_id": str(alumni["_id"]),
            "firstName": alumni.get("firstName"),
            "lastName": alumni.get("lastName"),
            "email": alumni.get("email"),
            "skills": alumni.get("skills"),
            "areaOfInterest": alumni.get("areaOfInterest"),
            "similarity": float(score)
        })

    return sorted(results, key=lambda x: x["similarity"], reverse=True)


# ================= ROOT TEST =================
@app.get("/")
def home():
    return {"message": "Alumni Suggestion API Running 🚀"}
