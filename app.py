from fastapi import FastAPI
from pydantic import BaseModel

from inference import generate

app = FastAPI()


class Prompt(BaseModel):
    prompt: str

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]


@app.get("/")
def home():
    return {
        "message": "MicroGPT API is running!"
    }


@app.post("/generate")
def generate_text(data: Prompt):
    return {
        "response": generate(data.prompt)
    }


@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [
            {
                "id": "micro-gpt",
                "object": "model",
                "owned_by": "Ihsan"
            }
        ]
    }



@app.post("/v1/chat/completions")
def chat(request: ChatRequest):

    prompt = request.messages[-1].content

    response = generate(prompt)

    return {
        "id": "micro-gpt-chat",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response
                },
                "finish_reason": "stop"
            }
        ]
    }
