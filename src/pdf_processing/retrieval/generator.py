import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # pdf_processing

from dotenv import load_dotenv
from groq import Groq

from config import ANSWER_MODEL, MAX_ANSWER_TOKENS
from tracing import traceable

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY is not set. Add GROQ_API_KEY=your_key to the .env file.")

client = Groq(api_key=api_key)


@traceable(name="llm", run_type="llm")
def generate_answer(prompt, model=ANSWER_MODEL, reasoning_effort=None, retries=3):
    settings = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_completion_tokens": MAX_ANSWER_TOKENS,
    }
    if reasoning_effort:
        settings["reasoning_effort"] = reasoning_effort

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(**settings)
            answer = response.choices[0].message.content or ""
            return answer.split("</think>")[-1].strip()   # drop thinking text if any

        except Exception as e:
            print("LLM error:", e)
            if "429" not in str(e):   # only rate limits are worth a retry
                break
            time.sleep((attempt + 1) * 15)

    return None
