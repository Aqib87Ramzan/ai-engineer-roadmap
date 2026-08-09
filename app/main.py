from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

# Explicitly point to the .env file next to this main.py, regardless of cwd
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
import os


# Initialize FastAPI app
app = FastAPI(title="AI Engineer Roadmap - Day 1")

# Initialize OpenAI client (works with OpenAI-compatible APIs too)
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    # base_url="https://api.groq.com/openai/v1"  # example: swap for any OpenAI-compatible provider
)

# ---- Request schema for /chat ----
class ChatRequest(BaseModel):
    message: str


# ---- /health endpoint ----
@app.get("/health")
def health_check():
    return {"status": "ok"}


# ---- /chat endpoint ----
@app.post("/chat")
def chat(request: ChatRequest):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # or any model your provider supports
            messages=[
                {"role": "user", "content": request.message}
            ]
        )
        reply = response.choices[0].message.content
        return {"response": reply}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))