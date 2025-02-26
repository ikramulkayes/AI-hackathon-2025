import streamlit as st
import requests
import json
from dotenv import load_dotenv
import os
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import re

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

# Configure Qdrant client
qdrant_client = QdrantClient(host=QDRANT_URL, port=QDRANT_PORT)

# Use a multilingual model for embeddings (we search in English)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Gemini translation function via OpenRouter (for input translation only)
def gemini_translate(text, source_lang, target_lang):
    """
    Translate text using Gemini via OpenRouter.
    """
    prompt = f"Translate the following text from {source_lang} to {target_lang}:\n\n{text}"
    messages = [{"role": "user", "content": prompt}]
    data = {
        "model": "deepseek/deepseek-r1-distill-llama-70b:free",
        "messages": messages,
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "<YOUR_SITE_URL>",
        "X-Title": "<YOUR_SITE_NAME>",
    }
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        data=json.dumps(data)
    )
    if response.status_code == 200:
        response_json = response.json()
        return response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
    else:
        st.error("Translation error.")
        return ""

def translate_to_english(text):
    return gemini_translate(text, source_lang="Bengali", target_lang="English")

# Define a prompt template for constitution comparison (for search/comparison queries)
constitution_prompt = PromptTemplate(
    input_variables=["article_info", "similar_articles"],
    template="""আপনি একজন সংবিধান বিশেষজ্ঞ। নিম্নলিখিত অনুচ্ছেদ এবং অন্যান্য দেশের অনুরূপ অনুচ্ছেদের তুলনা করুন।

সংবিধানের অনুচ্ছেদ:
{article_info}

অনুরূপ অনুচ্ছেদ:
{similar_articles}

সমস্ত অনুচ্ছেদ এবং তাদের সেকশন নম্বর উল্লেখ করুন এবং তুলনামূলক বিশ্লেষণ প্রদান করুন।
"""
)

# Define a prompt template for general conversation (non-search queries)
general_prompt = PromptTemplate(
    input_variables=["conversation_history", "latest_input"],
    template="""আপনি একজন সহায়ক সংবিধান বিশেষজ্ঞ। পূর্ববর্তী কথোপকথন এবং ব্যবহারকারীর সর্বশেষ ইনপুটের ভিত্তিতে একটি উত্তর তৈরি করুন।

কথোপকথনের ইতিহাস:
{conversation_history}

ব্যবহারকারীর সর্বশেষ ইনপুট:
{latest_input}

উত্তর (বাংলায়):
"""
)

# Use OpenRouter (DeepSeek) for generating the final response.
def openrouter_generate_response(prompt_messages):
    data = {
        "model": "deepseek/deepseek-r1-distill-llama-70b:free",
        "messages": prompt_messages,
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "<YOUR_SITE_URL>",
        "X-Title": "<YOUR_SITE_NAME>",
    }
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        data=json.dumps(data)
    )
    if response.status_code == 200:
        response_json = response.json()
        return response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
    else:
        st.error("Error generating response from DeepSeek.")
        return ""

# Chain functions
def generate_constitution_comparison(main_article, similar_articles):
    prompt = constitution_prompt.format(article_info=main_article, similar_articles=similar_articles)
    messages = [{"role": "user", "content": prompt}]
    return openrouter_generate_response(messages)

def generate_general_response(conversation_history, latest_input):
    prompt = general_prompt.format(conversation_history=conversation_history, latest_input=latest_input)
    messages = [{"role": "user", "content": prompt}]
    return openrouter_generate_response(messages)

# Search function: search in vector DB using English query
def search_constitution(query_text, country_filter=None):
    query_embedding = model.encode(query_text)
    query_filter = {"must": [{"key": "country", "match": {"value": country_filter}}]} if country_filter else None
    search_results = qdrant_client.search(
        collection_name="constitutions",
        query_vector=query_embedding.tolist(),
        query_filter=query_filter,
        limit=3
    )
    return [result.payload for result in search_results if result.score >= 0.4] if search_results else []

def is_search_query(user_input):
    search_keywords = ['অনুসন্ধান', 'প্রথম', 'আইন', 'অনুচ্ছেদ']
    return any(word in user_input for word in search_keywords) and extract_country(user_input)

def is_comparison_query(user_input):
    return "তুলনা" in user_input.lower()

# Updated countries list
COUNTRIES = ["বাংলাদেশ", "ফিনল্যান্ড", "সুইডেন"]
countrydic = {"বাংলাদেশ": "Bangladesh", "ফিনল্যান্ড": "Finland", "সুইডেন": "Sweden"}

def extract_country(user_input):
    for country in COUNTRIES:
        if re.search(rf'\b{country}\b', user_input):
            return countrydic[country]
    return None

# Streamlit App Setup
st.title("বাংলা সংবিধান তুলনা ও অনুসন্ধান চ্যাটবট")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "আপনি একজন সংবিধান বিশেষজ্ঞ।"},
        {"role": "assistant", "content": "স্বাগতম! সংবিধানের অনুচ্ছেদ সম্পর্কে জিজ্ঞাসা করুন।"}
    ]

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("আপনার প্রশ্ন লিখুন..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Translate user input from Bengali to English for searching
    translated_query = translate_to_english(prompt)
    
    # Initialize default values to avoid NameError
    main_article = "No matching article found."
    similar_articles = ""
    
    # Determine if this is a search/comparison query
    if is_search_query(prompt) or is_comparison_query(prompt):
        country = extract_country(prompt)
        print(f"Country: {country} translated_query: {translated_query}") 
        articles = search_constitution(translated_query, country_filter=country)
        if articles:
            main_article = articles[0].get('text', "No matching article found.")
            similar_articles = "\n\n".join(
                f"Country: {art.get('country', 'Unknown')}\nArticle: {art.get('text', '')}" 
                for art in articles[1:]
            )
            print(f"Main Article: {main_article}")
            print(f"Similar Articles: {similar_articles}")
        # Generate response using constitution comparison prompt
        response_text = generate_constitution_comparison(main_article, similar_articles)
    else:
        # Build conversation history from previous messages (excluding system)
        conversation_history = "\n".join(
            f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages if msg['role'] != "system"
        )
        response_text = generate_general_response(conversation_history, translated_query)
    
    with st.chat_message("assistant"):
        st.markdown(response_text)
    st.session_state.messages.append({"role": "assistant", "content": response_text})
