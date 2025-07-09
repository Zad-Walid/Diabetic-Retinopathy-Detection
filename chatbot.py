# chatbot.py
import requests
import re
from rag_engine import retrieve_relevant_chunks


# Replace with your real API key
TOGETHER_API_KEY = "tgp_v1_UuITIe7GLmZTAaJx9YudjDiUklsdG3vo10tdR7ZZ7xE"
API_URL = "https://api.together.xyz/v1/chat/completions"

def load_system_prompt():
    with open("system_prompt.txt", "r") as f:
        return f.read()

def clean_response(text):
    text = re.sub(r'^Bot:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'.*Fig.*?medical book.*\n?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'.*Figure.*\n?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'.*your medical book.*\n?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'.*reference[s]?\s*\d+.*\n?', '', text, flags=re.IGNORECASE)
    text = re.sub(r"Here'?s a simple breakdown.*\n?", '', text, flags=re.IGNORECASE)
    text = re.sub(r".*AI language model.*\n?", '', text, flags=re.IGNORECASE)
    text = re.sub(r'.*consult.*healthcare professional.*\n?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n{2,}', '\n\n', text)

    return text.strip()

def send_to_mixtral_with_rag(user_input):
    context = retrieve_relevant_chunks(user_input)
    system_prompt = load_system_prompt()

    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }

    messages = [
        {"role": "system", "content": system_prompt + f"\n\nContext from medical book:\n{context}"},
        {"role": "user", "content": user_input}
    ]

    payload = {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "messages": messages,
        "temperature": 0.5,
        "max_tokens": 512
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    data = response.json()

    if "choices" in data:
        raw_text = data["choices"][0]["message"]["content"]
        cleaned_text = clean_response(raw_text)
        return cleaned_text
    else:
        return f"Error: {data.get('error', 'No choices returned')}"