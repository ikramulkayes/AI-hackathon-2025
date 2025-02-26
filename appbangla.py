import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import re
from deep_translator import GoogleTranslator  # Translation module

# Load environment variables
load_dotenv()
api_key = os.getenv("API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

genai.configure(api_key=api_key)
chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=api_key)
memory = ConversationBufferMemory(return_messages=True)
qdrant_client = QdrantClient(host=QDRANT_URL, port=QDRANT_PORT)

# Use a multilingual model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

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

constitution_chain = LLMChain(llm=chat_model, prompt=constitution_prompt)

general_prompt = PromptTemplate(
    input_variables=["conversation_history", "latest_input"],
    template="""আপনি একজন সহায়ক সংবিধান বিশেষজ্ঞ। পূর্ববর্তী কথোপকথন এবং ব্যবহারকারীর সর্বশেষ ইনপুটের ভিত্তিতে একটি উত্তর তৈরি করুন।

কথোপকথনের ইতিহাস:
{conversation_history}

ব্যবহারকারীর সর্বশেষ ইনপুট:
{latest_input}

উত্তর:
"""
)
general_chain = LLMChain(llm=chat_model, prompt=general_prompt)

def translate_to_english(text):
    """Translate Bangla text to English."""
    return GoogleTranslator(source='bn', target='en').translate(text)

def translate_to_bangla(text):
    """Translate English text to Bangla."""
    return GoogleTranslator(source='en', target='bn').translate(text)

def search_constitution(query_text, country_filter=None):
    """Search for the translated query in the vector database."""
    query_embedding = model.encode(query_text)
    query_filter = {"must": [{"key": "country", "match": {"value": country_filter}}]} if country_filter else None
    search_results = qdrant_client.search(
        collection_name="constitutions", query_vector=query_embedding.tolist(), query_filter=query_filter, limit=3
    )
    return [result.payload for result in search_results if result.score >= 0.4] if search_results else []

def is_comparison_query(user_input):
    return "তুলনা" in user_input.lower()

def is_search_query(user_input):
    search_keywords = ['অনুসন্ধান', 'প্রথম', 'আইন', 'অনুচ্ছেদ']
    return any(word in user_input for word in search_keywords) and extract_country(user_input)

COUNTRIES = ["বাংলাদেশ", "ভারত", "যুক্তরাষ্ট্র", "জার্মানি", "ফ্রান্স", "ব্রাজিল"]

def extract_country(user_input):
    for country in COUNTRIES:
        if re.search(rf'\b{country}\b', user_input):
            return country
    return None

st.title("বাংলা সংবিধান তুলনা ও অনুসন্ধান চ্যাটবট")
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "আপনি একজন সংবিধান বিশেষজ্ঞ।"}]
    st.session_state.messages.append({"role": "assistant", "content": "স্বাগতম! সংবিধানের অনুচ্ছেদ সম্পর্কে জিজ্ঞাসা করুন।"})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("আপনার প্রশ্ন লিখুন..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response_text = ""
    translated_query = translate_to_english(prompt)  # Convert Bangla input to English

    if is_comparison_query(prompt):
        country = extract_country(prompt)
        articles = search_constitution(translated_query, country_filter=country)
        main_article = articles[0]['text'] if articles else "No matching articles found."
        similar_articles = "\n\n".join(f"Country: {art.get('country', 'Unknown')}\nArticle: {art.get('text', '')}" for art in articles[1:])
        response_data = constitution_chain.invoke({"article_info": main_article, "similar_articles": similar_articles})
        response_text = response_data.get('text', "Sorry, comparison could not be generated.")
    elif is_search_query(prompt):
        country = extract_country(prompt)
        articles = search_constitution(translated_query, country_filter=country)
        response_text = "\n\n".join(f"Country: {art.get('country', 'Unknown')}\nArticle: {art.get('text', '')}" for art in articles) if articles else "No articles found."
    else:
        conversation_history = "\n".join(f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages if msg['role'] != "system")
        response_data = general_chain.invoke({"conversation_history": conversation_history, "latest_input": translated_query})
        response_text = response_data.get('text', "Sorry, I didn't understand.")

    translated_response = translate_to_bangla(response_text)  # Convert response back to Bangla

    with st.chat_message("assistant"):
        st.markdown(translated_response)
    st.session_state.messages.append({"role": "assistant", "content": translated_response})
    memory.chat_memory.add_user_message(prompt)
    memory.chat_memory.add_ai_message(translated_response)
