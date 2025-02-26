import streamlit as st
import requests
import json
from dotenv import load_dotenv
import os
import re
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

# Configure Qdrant client
qdrant_client = QdrantClient(host=QDRANT_URL, port=QDRANT_PORT)

# Use a multilingual model for embeddings (search in English)
model = SentenceTransformer('all-MiniLM-L6-v2')

# OpenRouter API call function
def query_openrouter(prompt):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek/deepseek-r1-distill-llama-70b:free",
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return "Sorry, an error occurred while fetching the response."

# Define Prompts
constitution_prompt = PromptTemplate(
    input_variables=["article_info", "similar_articles"],
    template="""আপনি একজন সংবিধান বিশেষজ্ঞ। নিম্নলিখিত অনুচ্ছেদ এবং অন্যান্য দেশের অনুরূপ অনুচ্ছেদের তুলনা করুন।
    
সংবিধানের অনুচ্ছেদ:
{article_info}
    
অনুরূপ অনুচ্ছেদ:
{similar_articles}
    
সমস্ত অনুচ্ছেদ এবং তাদের সেকশন নম্বর উল্লেখ করুন এবং তুলনামূলক বিশ্লেষণ প্রদান করুন।"""
)

general_prompt = PromptTemplate(
    input_variables=["conversation_history", "latest_input"],
    template="""আপনি একজন সহায়ক সংবিধান বিশেষজ্ঞ। পূর্ববর্তী কথোপকথন এবং ব্যবহারকারীর সর্বশেষ ইনপুটের ভিত্তিতে একটি উত্তর তৈরি করুন।
    
কথোপকথনের ইতিহাস:
{conversation_history}
    
ব্যবহারকারীর সর্বশেষ ইনপুট:
{latest_input}
    
উত্তর:"""
)

# Search function: search in vector DB using English query
def search_constitution(query_text, country_filter=None):
    query_embedding = model.encode(query_text)
    query_filter = {"must": [{"key": "country", "match": {"value": country_filter}}]} if country_filter else None
    search_results = qdrant_client.search(
        collection_name="constitutions",
        query_vector=query_embedding.tolist(),
        query_filter=query_filter,
        limit=5
    )
    return [result.payload for result in search_results if result.score >= 0.4] if search_results else []

# Query classification functions
def is_search_query(user_input):
    search_keywords = ['অনুসন্ধান', 'প্রথম', 'আইন', 'অনুচ্ছেদ']
    return any(word in user_input for word in search_keywords) and extract_country(user_input)

def is_comparison_query(user_input):
    return "তুলনা" in user_input

# Country mapping
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
    
    # Initialize response
    response_text = ""
    
    # Search Query Handler
    if is_search_query(prompt):
        country = extract_country(prompt)
        articles = search_constitution(prompt, country_filter=country)
        response_text = "\n\n".join(
            f"Country: {art.get('country', 'Unknown')}\nArticle: {art.get('text', '')}" 
            for art in articles
        ) if articles else "কোনো মিলিত অনুচ্ছেদ পাওয়া যায়নি।"
    
    # Comparison Query Handler
    elif is_comparison_query(prompt):
        country = extract_country(prompt)
        main_articles = search_constitution(prompt, country_filter=country)
        main_article_info = main_articles[0].get('text', "") if main_articles else ""
        similar_articles = search_constitution(prompt)
        similar_articles_filtered = [art for art in similar_articles if art.get('country') != country][:2]
        similar_articles_text = "\n\n".join(
            f"Country: {art.get('country', 'Unknown')}\nArticle: {art.get('text', '')}"
            for art in similar_articles_filtered
        )
        comparison_input = constitution_prompt.format(article_info=main_article_info, similar_articles=similar_articles_text)
        response_text = query_openrouter(comparison_input)
    
    # General Query Handler
    else:
        conversation_history = "\n".join(
            f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages if msg['role'] != "system"
        )
        general_input = general_prompt.format(conversation_history=conversation_history, latest_input=prompt)
        response_text = query_openrouter(general_input)
    
    with st.chat_message("assistant"):
        st.markdown(response_text)
    st.session_state.messages.append({"role": "assistant", "content": response_text})
