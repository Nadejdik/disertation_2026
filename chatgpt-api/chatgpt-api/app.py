from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
import os
import time
import logging
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging for research
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatgpt_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="ChatGPT API", description="Research-friendly ChatGPT API")
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Pydantic models
class Question(BaseModel):
    prompt: str = Field(..., description="The question or prompt to send to ChatGPT")
    model: Optional[str] = Field("gpt-4o-mini", description="OpenAI model to use")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens in response")

class Message(BaseModel):
    role: str = Field(..., description="Role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message content")

class Conversation(BaseModel):
    messages: List[Message] = Field(..., description="List of conversation messages")
    model: Optional[str] = Field("gpt-4o-mini", description="OpenAI model to use")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None)

class ChatResponse(BaseModel):
    answer: str
    model: str
    usage: dict
    latency_ms: float
    cost_estimate: float

# Cost per 1M tokens (approximate, update as needed)
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
}

def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate estimated cost for API call"""
    if model not in PRICING:
        return 0.0
    
    pricing = PRICING[model]
    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost

@app.post("/chat", response_model=ChatResponse)
def chat(q: Question):
    """
    Send a single prompt to ChatGPT and get a response.
    Perfect for simple Q&A and research experiments.
    """
    try:
        start_time = time.time()
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model=q.model,
            messages=[{"role": "user", "content": q.prompt}],
            temperature=q.temperature,
            max_tokens=q.max_tokens
        )
        
        latency = (time.time() - start_time) * 1000  # Convert to ms
        
        # Extract response
        answer = response.choices[0].message.content
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
        
        # Calculate cost
        cost = calculate_cost(
            q.model,
            response.usage.prompt_tokens,
            response.usage.completion_tokens
        )
        
        # Log for research
        logger.info(f"Model: {q.model} | Tokens: {usage['total_tokens']} | "
                   f"Latency: {latency:.2f}ms | Cost: ${cost:.6f}")
        
        return ChatResponse(
            answer=answer,
            model=q.model,
            usage=usage,
            latency_ms=round(latency, 2),
            cost_estimate=round(cost, 6)
        )
    
    except Exception as e:
        logger.error(f"Error in /chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/conversation", response_model=ChatResponse)
def conversation(conv: Conversation):
    """
    Send a full conversation history to ChatGPT.
    Useful for multi-turn dialogues and context-aware responses.
    """
    try:
        start_time = time.time()
        
        # Convert Pydantic models to dict
        messages = [{"role": msg.role, "content": msg.content} for msg in conv.messages]
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model=conv.model,
            messages=messages,
            temperature=conv.temperature,
            max_tokens=conv.max_tokens
        )
        
        latency = (time.time() - start_time) * 1000
        
        answer = response.choices[0].message.content
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
        
        cost = calculate_cost(
            conv.model,
            response.usage.prompt_tokens,
            response.usage.completion_tokens
        )
        
        logger.info(f"Model: {conv.model} | Tokens: {usage['total_tokens']} | "
                   f"Latency: {latency:.2f}ms | Cost: ${cost:.6f}")
        
        return ChatResponse(
            answer=answer,
            model=conv.model,
            usage=usage,
            latency_ms=round(latency, 2),
            cost_estimate=round(cost, 6)
        )
    
    except Exception as e:
        logger.error(f"Error in /chat/conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "healthy", "api": "ChatGPT API"}

@app.get("/")
def root():
    """Root endpoint - redirects to docs"""
    return {
        "message": "ChatGPT API is running",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
