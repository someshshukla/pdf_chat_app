import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY not found in .env file.")
    exit()
else:
    print(f"Using GOOGLE_API_KEY starting with: {GOOGLE_API_KEY[:5]}...")

try:
    genai.configure(api_key=GOOGLE_API_KEY)
    print("\nAvailable models supporting 'generateContent':")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"\nAn error occurred: {e}")