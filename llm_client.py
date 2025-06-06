import httpx
import json

LLM_API_URL = "http://127.0.0.1:1234/v1/chat/completions"
HEADERS = {
    "Content-Type": "application/json"
}

async def main_llm(messages: list) -> str:
    payload = {
        "model": "mistralai/mathstral-7b-v0.1",
        "messages": messages,
        "stream": False,
        "max_tokens": 512,
        "temperature": 0.2,
        "top_p": 0.9,
        "presence_penalty": 0.1,
    }

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(LLM_API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content']