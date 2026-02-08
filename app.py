from fastapi import FastAPI
from pydantic import BaseModel
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import faiss
import os
from dotenv import load_dotenv
import numpy as np

app = FastAPI()
load_dotenv() 

MONGO_URI = os.getenv("MONGO_URI")

# MongoDB
MONGO_URI = "mongodb+srv://srimaniram_db_user:jothir1234@cluster0.nsmhmzs.mongodb.net/?appName=Cluster0"
client = MongoClient(MONGO_URI)

# 🔍 CONNECTION CHECK
try:
    client.admin.command("ping")
    print("✅ MongoDB connected successfully (FastAPI)")
except Exception as e:
    print("❌ MongoDB connection failed:", e)

db = client["test"]
signups = db["signups"]

# ================= ML MODEL =================
model = SentenceTransformer("all-mpnet-base-v2")

# ================= REQUEST MODEL =================
class SuggestRequest(BaseModel):
    studentSkills: str
    studentAoi: str

# ================= API ENDPOINT =================
@app.post("/suggest")
def suggest_alumni(req: SuggestRequest):
    print("📥 Incoming request:", req)

    # 🔍 TEMP: Fetch ALL documents to confirm data exists
    alumni_list = list(signups.find({"role": "alumni"}))
    print("📊 Total documents in signups:", len(alumni_list))

    # 🔍 TEMP: Print role values (first 5 docs)
    for a in alumni_list[:5]:
        print("👤 Role value:", a.get("role"))

    if not alumni_list:
        return []

    # ================= EMBEDDINGS =================
    alumni_texts = [
        f"{a.get('skills','')} {a.get('areaOfInterest','')} {a.get('about','')}"
        for a in alumni_list
    ]

    alumni_embeddings = model.encode(alumni_texts, convert_to_numpy=True)
    faiss.normalize_L2(alumni_embeddings)

    query_vec = model.encode(
        [f"{req.studentSkills} {req.studentAoi}"],
        convert_to_numpy=True
    )
    faiss.normalize_L2(query_vec)

    index = faiss.IndexFlatIP(alumni_embeddings.shape[1])
    index.add(alumni_embeddings)

    D, I = index.search(query_vec, len(alumni_list))

    # ================= RESULTS =================
    threshold = 0.15
    results = []

    for i, score in zip(I[0], D[0]):
        if score < threshold:
            continue

        alumni = alumni_list[i]
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