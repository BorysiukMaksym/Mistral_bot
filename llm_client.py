import httpx
import json

LLM_API_URL = "http://127.0.0.1:1234/v1/completions"
HEADERS = {
    "Content-Type": "application/json"
}


def convert_messages_to_prompt(messages: list[dict]) -> str:
    prompt = ""
    for m in messages:
        role = m["role"]
        if role == "system":
            prompt += f"[System]: {m['content']}\n"
        elif role == "user":
            prompt += f"[User]: {m['content']}\n"
        elif role == "assistant":
            prompt += f"[Assistant]: {m['content']}\n"
    prompt += "[Assistant]:"
    return prompt


async def main_llm(messages: list[dict]) -> str:
    prompt = convert_messages_to_prompt(messages)

    payload = {
        "model": "mistralai/mistral-7b-instruct-v0.3",
        "prompt": prompt,
        "max_tokens": 512,
        "temperature": 0.2,
        "top_p": 0.9,
        "presence_penalty": 0.1,
    }

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(LLM_API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['text'].strip()
