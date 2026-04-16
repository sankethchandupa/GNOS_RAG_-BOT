from fastapi import FastAPI
from pydantic import BaseModel
from rag_bot import rag_pipeline

app = FastAPI()

# -------------------------
# REQUEST FORMAT
# -------------------------
class Question(BaseModel):
    query: str

# -------------------------
# LOAD YOUR EXISTING BOT CODE HERE
# (IMPORTANT: paste only needed functions)
# -------------------------

@app.post("/ask")
def ask(q: Question):

    question = q.query   # ✅ FIXED

    answer = rag_pipeline(question)

    return {"answer": answer}