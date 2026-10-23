
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


app = FastAPI(
    title="ZentroMart AI Support API",
    description="API for the ZentroMart AI customer support agent",
    version="1.0.0",
)

# Development ke liye frontend origins.
# Production mein apne actual frontend domain tak restrict karna.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    answer: str
    status: str


@app.get("/")
def home():
    return {
        "message": "ZentroMart AI Support API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "ZentroMart AI Support API"
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    # Temporary integration point:
    # Yahan existing RAG agent ka actual function call karna hai.
    raise HTTPException(
        status_code=501,
        detail=(
            "FastAPI is ready, but the existing RAG "
            "agent function must be connected."
        ),
    )
