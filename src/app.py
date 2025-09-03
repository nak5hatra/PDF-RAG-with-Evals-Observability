from fastapi import FastAPI
from pydantic import BaseModel
from .rag_chain import rag_chain
import time

app = FastAPI(title="PDF RAG API", version="1.0.0")


class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    latency: float


@app.post("/ask", response_model=AnswerResponse)
def ask_question(payload: QuestionRequest):
    start = time.time()
    result = rag_chain.invoke({ # type: ignore
        "question": payload.question,
        "context": [],
        "answer": ""
    })
    end = time.time()
    latency = round(end - start, 2)
    
    return AnswerResponse(
        answer=result["answer"], # type: ignore
        latency=latency
    )