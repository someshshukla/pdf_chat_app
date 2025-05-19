import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY not found in .env file.")
    print("Please ensure your .env file is in the same directory and contains: GOOGLE_API_KEY='YOUR_KEY'")
    exit()
else:
    print(f"Using GOOGLE_API_KEY starting with: {GOOGLE_API_KEY[:5]}...")

try:
    genai.configure(api_key=GOOGLE_API_KEY)

    print("\nAvailable models supporting 'generateContent' (for chat):")
    chat_models_found = []
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
            chat_models_found.append(m.name)
    if not chat_models_found:
        print("  No models found supporting 'generateContent'. Check API key permissions.")

    print("\n----------------------------------------------------")
    print("Available models supporting 'embedContent' (for embeddings):")
    found_embedding_model_001 = False
    embedding_models_found = []
    for m in genai.list_models():
        if 'embedContent' in m.supported_generation_methods:
            print(f"- {m.name}")
            embedding_models_found.append(m.name)
            if m.name == "models/embedding-001":
                found_embedding_model_001 = True
    
    if not embedding_models_found:
        print("  No models found supporting 'embedContent'. Check API key permissions.")
    
    if found_embedding_model_001:
        print("\n>>> 'models/embedding-001' IS available for embeddings with this API key.")
    else:
        print("\n>>> WARNING: 'models/embedding-001' was NOT found for embeddings. Please check the list above for a suitable alternative embedding model if needed, or verify API key permissions for this model.")

except Exception as e:
    print(f"\nAn error occurred while trying to list models: {e}")
    print("This could be due to an invalid GOOGLE_API_KEY, the 'Generative Language API' not being enabled for your project in Google Cloud Console, or billing issues.")