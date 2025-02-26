# import google.generativeai as genai
from dotenv import load_dotenv
import os

# # Load environment variables from .env file
# load_dotenv()

# # Access the API key from the .env file
# api_key = os.getenv("API_KEY")

# # Configure the generative AI client
# genai.configure(api_key=api_key)
# model = genai.GenerativeModel('gemini-1.5-flash')
# # Generate text using the Gemini model
# response = model.generate_content("Write a story about an AI and magic")
# print(response.text)



import google.generativeai as genai

# Used to securely store your API key
load_dotenv()

api_key = os.getenv("API_KEY")


genai.configure(api_key=api_key)
for m in genai.list_models():
#   print(m.name)
  if 'embedContent' in m.supported_generation_methods:
    print(m.name)
