
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.rag_agent import respond

app = FastAPI(title="ZentroMart AI Support API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5174",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    status: str


@app.get("/")
def home():
    return {"message": "ZentroMart AI Support API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Please enter a question."
        )

    try:
        answer = respond(question)

        return ChatResponse(
            answer=answer,
            status="success"
        )

    except Exception as exc:
        print(f"Chat error: {exc}")
        raise HTTPException(
            status_code=500,
            detail="The AI support service could not generate a response. Check the backend logs."
        ) from exc
