import os
import time
import random
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


PRICE_EXTRACT_SYSTEM = "You extract prices from negotiation transcripts."

PRICE_EXTRACT_PROMPT = (
    "Analyze the negotiation transcript. If a final deal was agreed upon, "
    "respond with ONLY the number (e.g. 850). If no deal was reached, respond with NONE."
    "\n\nTranscript:\n{transcript}"
)


def _retry(fn, max_retries=6):
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = min(2 ** attempt, 30)
            print(f"  API error (attempt {attempt+1}/{max_retries}): {e}, retrying in {wait}s...", flush=True)
            time.sleep(wait)


def _parse_price(resp):
    resp = resp.strip().upper()
    if resp == "NONE" or "none" in resp.lower():
        return None
    try:
        return float(resp.replace("$", "").replace(",", ""))
    except ValueError:
        return None


class OpenAIModel:
    def __init__(self, model_id):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ["OPEN_AI_API_KEY"], timeout=60)
        self.model_id = model_id
        self.name = model_id

    def chat(self, system, user):
        resp = _retry(lambda: self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        ))
        return resp.choices[0].message.content

    def extract_price(self, transcript):
        return _parse_price(self.chat(PRICE_EXTRACT_SYSTEM, PRICE_EXTRACT_PROMPT.format(transcript=transcript)))


class GeminiModel:
    def __init__(self, model_id):
        from google import genai
        from google.genai import types
        self.client = genai.Client(
            api_key=os.environ["GOOGLE_API_KEY"],
            http_options=types.HttpOptions(timeout=60_000),
        )
        self.model_id = model_id
        self.name = model_id

    def chat(self, system, user):
        from google.genai import types
        resp = _retry(lambda: self.client.models.generate_content(
            model=self.model_id,
            contents=user,
            config=types.GenerateContentConfig(system_instruction=system),
        ))
        return resp.text

    def extract_price(self, transcript):
        return _parse_price(self.chat(PRICE_EXTRACT_SYSTEM, PRICE_EXTRACT_PROMPT.format(transcript=transcript)))


class GroqModel:
    def __init__(self, model_id):
        from groq import Groq
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"], timeout=60)
        self.model_id = model_id
        self.name = model_id

    def chat(self, system, user):
        resp = _retry(lambda: self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        ))
        return resp.choices[0].message.content

    def extract_price(self, transcript):
        return _parse_price(self.chat(PRICE_EXTRACT_SYSTEM, PRICE_EXTRACT_PROMPT.format(transcript=transcript)))


gemini_3_1_flash_lite = GeminiModel("gemini-3.1-flash-lite-preview")
gemini_2_5_flash      = GeminiModel("gemini-2.5-flash")
gpt_oss_120b          = GroqModel("openai/gpt-oss-120b")

eval_model = gemini_3_1_flash_lite


def get_gen_roster():
    return [
        (OpenAIModel("gpt-5.4"),                 0.125),
        (GeminiModel("gemini-3.1-pro-preview"),  0.125),
        (GeminiModel("gemini-3-flash-preview"),   0.45),
        (OpenAIModel("gpt-5.4-mini"),             0.30),
    ]


GRPO_OPPONENT_ROSTER = [
    (gemini_3_1_flash_lite, 0.45),
    (gemini_2_5_flash,      0.275),
    (gpt_oss_120b,          0.275),
]

PPO_OPPONENT_ROSTER = [
    (gemini_3_1_flash_lite, 0.45),
    (gemini_2_5_flash,      0.30),
    (gpt_oss_120b,          0.25),
]


class FallbackModel:
    def __init__(self, primary, fallbacks):
        self.primary = primary
        self.fallbacks = fallbacks
        self.name = primary.name

    def chat(self, system, user):
        try:
            return self.primary.chat(system, user)
        except Exception:
            print(f"  {self.primary.name} failed, falling back...", flush=True)
            for fb in self.fallbacks:
                try:
                    return fb.chat(system, user)
                except Exception:
                    print(f"  {fb.name} also failed, trying next...", flush=True)
            raise

    def extract_price(self, transcript):
        try:
            return self.primary.extract_price(transcript)
        except Exception:
            print(f"  {self.primary.name} extract_price failed, falling back...", flush=True)
            for fb in self.fallbacks:
                try:
                    return fb.extract_price(transcript)
                except Exception:
                    continue
            raise


def pick_from_roster(roster):
    models, weights = zip(*roster)
    primary = random.choices(models, weights=weights, k=1)[0]
    fallbacks = [m for m in models if m is not primary]
    return FallbackModel(primary, fallbacks)
