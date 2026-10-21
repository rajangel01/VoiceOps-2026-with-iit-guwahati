
import os
import requests
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_URL = "http://localhost:11434/api/chat"

SYSTEM_PROMPT = """
You are ZentroMart's AI customer support assistant.
ZentroMart is a fictional Indian online store selling groceries
and electronics.

Rules:
1. Introduce yourself as an AI assistant in your first reply.
2. Be polite, concise, and helpful.
3. Never invent order details, policies, dates, or refund amounts.
4. Company policy documents and order tools are not connected yet.
5. If you do not know something, clearly say so.
6. Never claim to have checked an order or created a ticket.
"""

def respond(message, history=None):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": message})

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": messages,
            "stream": False
        },
        timeout=120
    )

    response.raise_for_status()
    data = response.json()
    return data["message"]["content"]
