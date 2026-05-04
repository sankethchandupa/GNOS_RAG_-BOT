from fastapi import FastAPI
from pydantic import BaseModel
from rag_bot import rag_pipeline

app = FastAPI()


class Question(BaseModel):
    query: str



@app.post("/ask")
def ask(q: Question):

    question = q.query   

    answer = rag_pipeline(question)

    return {"answer": answer}