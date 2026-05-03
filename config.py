import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PDF_FOLDER = "cambridge_cs_evaluator/data/mark_schemes"
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"