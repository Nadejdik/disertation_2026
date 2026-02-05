# ChatGPT API

A research-friendly FastAPI server that talks to OpenAI's ChatGPT. Perfect for dissertations, LLM comparisons, and experiments!

## What You Need First

1. **OpenAI API Key** - Get it from [platform.openai.com](https://platform.openai.com/api-keys)
2. **Python** - Version 3.8 or higher
3. **Internet** - For API calls

## Setup (3 steps)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Create `.env` file and add your API key
Copy `.env.example` to `.env`:
```bash
copy .env.example .env
```

Then edit `.env` and add your real API key:
```
OPENAI_API_KEY=sk-your-actual-key-here
```

**Never put API key in frontend code!**

### 3. Run the API
```bash
python app.py
```

Or with uvicorn:
```bash
uvicorn app:app --reload
```

API runs on: **http://127.0.0.1:8000**

**Interactive docs**: **http://127.0.0.1:8000/docs**

## API Endpoints

### POST /chat
Send a single prompt to ChatGPT (most common use case)

**Request:**
```json
{
  "prompt": "Explain LLMs in simple words",
  "model": "gpt-4o-mini",
  "temperature": 0.7,
  "max_tokens": 500
}
```

**Response:**
```json
{
  "answer": "Large Language Models (LLMs) are...",
  "model": "gpt-4o-mini",
  "usage": {
    "prompt_tokens": 8,
    "completion_tokens": 120,
    "total_tokens": 128
  },
  "latency_ms": 1234.56,
  "cost_estimate": 0.000123
}
```

### POST /chat/conversation
Send full conversation history (for multi-turn chat)

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
    {"role": "user", "content": "What about 2+3?"}
  ],
  "model": "gpt-4o-mini",
  "temperature": 0.7
}
```

### GET /health
Health check endpoint

### GET /
Root endpoint with API info

## Testing Your API

### Option 1: Interactive Docs (Easiest!)
1. Go to http://127.0.0.1:8000/docs
2. Click **POST /chat**
3. Click **Try it out**
4. Enter:
```json
{
  "prompt": "Hello, who are you?"
}
```
5. Click **Execute**

### Option 2: cURL
```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"Explain quantum computing\"}"
```

### Option 3: Python
```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/chat",
    json={"prompt": "What are LLMs?"}
)

print(response.json()["answer"])
```

### Option 4: JavaScript
```javascript
fetch("http://127.0.0.1:8000/chat", {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify({
    prompt: "What are LLMs?"
  })
})
.then(r => r.json())
.then(data => console.log(data.answer));
```

## For Research / Dissertation

This API includes features perfect for LLM research:

- **Automatic logging** - All requests logged to `chatgpt_api.log`
- **Latency tracking** - Measure response time
- **Cost estimation** - Track API costs per request
- **Token usage** - Monitor prompt/completion tokens
- **Model swapping** - Easy to compare models
- **Error handling** - Proper HTTP error codes

### Comparing Models
```python
models = ["gpt-4o-mini", "gpt-4o", "gpt-4"]
results = []

for model in models:
    response = requests.post(
        "http://127.0.0.1:8000/chat",
        json={
            "prompt": "Explain AI in 50 words",
            "model": model
        }
    )
    results.append(response.json())

# Analyze latency, cost, quality
```

## Common Mistakes to Avoid

- Putting API key in frontend code - Always keep it in backend `.env` file
- Not handling errors - This API returns proper HTTP status codes
- Sending very long prompts - Use `max_tokens` to control response length
- Forgetting rate limits - Check OpenAI's rate limits for your tier

## Available Models

- `gpt-4o-mini` - Fastest, cheapest (recommended for testing)
- `gpt-4o` - Balanced performance
- `gpt-4` - Most capable, but expensive

## Cost Tracking

The API automatically estimates costs based on:
- Input tokens (prompt)
- Output tokens (completion)
- Current OpenAI pricing

Check `chatgpt_api.log` for detailed logs!

## Configuration

Edit `app.py` to:
- Change default model
- Adjust temperature
- Update pricing
- Add custom logging

## License

MIT - Use freely for research and projects!
