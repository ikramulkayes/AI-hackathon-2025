import streamlit as st
import requests
import json
from dotenv import load_dotenv
import os
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

# Configure Qdrant client
try:
    qdrant_client = QdrantClient(host=QDRANT_URL, port=QDRANT_PORT)
    logger.info("Connected to Qdrant successfully")
except Exception as e:
    logger.error(f"Failed to connect to Qdrant: {e}")
    st.error("Failed to connect to vector database. Please check your configuration.")

# Use a multilingual model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Constants
COUNTRIES = {"বাংলাদেশ": "Bangladesh", "ফিনল্যান্ড": "Finland", "সুইডেন": "Sweden"}
MODEL_NAME = "deepseek/deepseek-r1-distill-llama-70b:free"  # Can be easily updated in one place

def make_openrouter_request(messages, temperature=0.7):
    """
    Make a request to the OpenRouter API.
    
    Args:
        messages (list): List of message dictionaries
        temperature (float, optional): Model temperature. Defaults to 0.7.
        
    Returns:
        str: Response text or error message
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://constitution-comparison-app.com",  # Update with your site
        "X-Title": "Bengali Constitution Comparison",
    }
    
    data = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
    }
    
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(data),
            timeout=30  # Add timeout
        )
        response.raise_for_status()
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    except requests.exceptions.RequestException as e:
        logger.error(f"OpenRouter API request failed: {e}")
        return f"Error generating response: {str(e)}"

def gemini_translate(text, source_lang, target_lang):
    """
    Use Gemini (via OpenRouter) for translation.
    
    Args:
        text (str): Text to translate
        source_lang (str): Source language
        target_lang (str): Target language
        
    Returns:
        str: Translated text
    """
    if not text.strip():
        return ""
        
    prompt = f"Translate the following text from {source_lang} to {target_lang}. Maintain the original formatting and preserve any special characters or technical terms:\n\n{text}"
    messages = [{"role": "user", "content": prompt}]
    
    logger.info(f"Translating from {source_lang} to {target_lang}")
    return make_openrouter_request(messages)

def translate_to_english(text):
    """Translate Bengali text to English."""
    return gemini_translate(text, source_lang="Bengali", target_lang="English")

def translate_to_bangla(text):
    """Translate English text to Bengali."""
    return gemini_translate(text, source_lang="English", target_lang="Bengali")

# Define prompts for constitutional comparison and general conversation
constitution_prompt = """আপনি একজন সংবিধান বিশেষজ্ঞ। নিম্নলিখিত অনুচ্ছেদ এবং অন্যান্য দেশের অনুরূপ অনুচ্ছেদের তুলনা করুন।

সংবিধানের অনুচ্ছেদ:
{article_info}

অনুরূপ অনুচ্ছেদ:
{similar_articles}

আপনার উত্তর নিম্নলিখিত বিন্যাসে দিন:

তুলনামূলক বিশ্লেষণ: [প্রথম দেশ]ের সংবিধানের অনুচ্ছেদ [নম্বর]: বিষয়বস্তু: [সংক্ষিপ্ত বিবরণ]। বিশেষ বৈশিষ্ট্য: [উল্লেখযোগ্য দিক]। 

[দ্বিতীয় দেশ]ের সংবিধানের অনুরূপ অনুচ্ছেদ: বিষয়বস্তু: [সংক্ষিপ্ত বিবরণ]। বিশেষ বৈশিষ্ট্য: [উল্লেখযোগ্য দিক]।

তুলনা ও বিশ্লেষণ: 
বিষয়বস্তু: [দুই দেশের অনুচ্ছেদের বিষয়বস্তুর তুলনা]
সংবিধানের কাঠামো: [দুই দেশের সংবিধানের কাঠামোগত পার্থক্য]
সাংস্কৃতিক ও ঐতিহাসিক প্রভাব: [সাংস্কৃতিক প্রভাব]
আইনি কাঠামো: [আইনি কাঠামো ও প্রয়োগের পার্থক্য]
উপসংহার: [সামগ্রিক বিশ্লেষণ ও উপসংহার]
"""

general_prompt = """আপনি একজন সহায়ক সংবিধান বিশেষজ্ঞ। আপনি সবসময় বাংলায় উত্তর দেবেন। পূর্ববর্তী কথোপকথন এবং ব্যবহারকারীর সর্বশেষ ইনপুটের ভিত্তিতে একটি উত্তর তৈরি করুন।

কথোপকথনের ইতিহাস:
{conversation_history}

ব্যবহারকারীর সর্বশেষ ইনপুট:
{latest_input}

আপনার উত্তর বাংলায় দিন। এই উত্তর সরাসরি ব্যবহারকারীকে দেখানো হবে, কোনো অতিরিক্ত অনুবাদ ছাড়াই। আপনার উত্তর বিস্তারিত, তথ্যপূর্ণ এবং সংবিধান বিশেষজ্ঞের মতো হওয়া উচিত। সবসময় বাংলায় উত্তর দিন।
"""

def generate_constitution_comparison(main_article, similar_articles):
    """
    Generate a comparison between constitutional articles.
    
    Args:
        main_article (str): The main article to compare
        similar_articles (str): Similar articles to compare with
        
    Returns:
        str: Comparison analysis
    """
    prompt = constitution_prompt.format(article_info=main_article, similar_articles=similar_articles)
    messages = [{"role": "user", "content": prompt}]
    return make_openrouter_request(messages)

def generate_general_response(conversation_history, latest_input):
    """
    Generate a general response based on conversation history.
    
    Args:
        conversation_history (str): Previous conversation
        latest_input (str): Latest user input
        
    Returns:
        str: Generated response
    """
    prompt = general_prompt.format(conversation_history=conversation_history, latest_input=latest_input)
    messages = [{"role": "user", "content": prompt}]
    
    # Important - this response is directly in Bengali, no translation needed
    return make_openrouter_request(messages)

def search_constitution(query_text, country_filter=None):
    """
    Search the constitution database using vector similarity.
    
    Args:
        query_text (str): The query text
        country_filter (str, optional): Country to filter results by
        
    Returns:
        list: List of matching articles
    """
    try:
        query_embedding = model.encode(query_text)
        query_filter = {"must": [{"key": "country", "match": {"value": country_filter}}]} if country_filter else None
        
        search_results = qdrant_client.search(
            collection_name="constitutions",
            query_vector=query_embedding.tolist(),
            query_filter=query_filter,
            limit=5  # Increase from 3 to 5 for more comprehensive results
        )
        
        results = [result.payload for result in search_results if result.score >= 0.4]
        logger.info(f"Found {len(results)} search results for query: {query_text}")
        return results
    except Exception as e:
        logger.error(f"Search error: {e}")
        return []

def is_comparison_query(user_input):
    """Check if the query is about comparing constitutional articles."""
    comparison_keywords = ["তুলনা", "পার্থক্য", "মিল", "সাদৃশ্য", "বিশ্লেষণ"]
    return any(keyword in user_input.lower() for keyword in comparison_keywords)

def is_search_query(user_input):
    """Check if the query is a search for specific articles."""
    search_keywords = ['অনুসন্ধান', 'খোঁজ', 'অনুচ্ছেদ', 'ধারা', 'আইন', 'আর্টিকেল']
    return (any(word in user_input.lower() for word in search_keywords) or 
            re.search(r'[০-৯]+\s*(?:নং|নম্বর)?', user_input) or  # Bengali number followed by "number" word
            any(country in user_input for country in COUNTRIES.keys()))

def extract_country(user_input):
    """Extract country name from user input."""
    for bangla_country, english_country in COUNTRIES.items():
        if re.search(rf'\b{bangla_country}\b', user_input):
            return english_country
    return None

def format_article(article):
    """Format article data for display."""
    country = article.get('country', 'Unknown')
    article_number = article.get('article_number', 'Unknown')
    text = article.get('text', '')
    
    # Translate country names back to Bengali
    country_translations = {v: k for k, v in COUNTRIES.items()}
    country_display = country_translations.get(country, country)
    
    return f"**দেশ:** {country_display}\n**অনুচ্ছেদ নং:** {article_number}\n\n{text}"

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="বাংলা সংবিধান তুলনা ও অনুসন্ধান চ্যাটবট",
        page_icon="📜",
        layout="wide"
    )
    
    # Add CSS for better appearance
    
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.markdown("<div class='main-header'><h1>📜 বাংলা সংবিধান তুলনা ও অনুসন্ধান চ্যাটবট</h1></div>", unsafe_allow_html=True)
        st.markdown("<div class='sub-header'><h4>🔍 বাংলাদেশ, ফিনল্যান্ড এবং সুইডেনের সংবিধান অনুসন্ধান ও তুলনা করুন</h4></div>", unsafe_allow_html=True)
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": "আপনি একজন সংবিধান বিশেষজ্ঞ।"},
            {"role": "assistant", "content": "স্বাগতম! আমি আপনাকে সংবিধান সম্পর্কে সাহায্য করতে পারি। আপনি যেকোনো দেশের (বাংলাদেশ, ফিনল্যান্ড, সুইডেন) সংবিধানের অনুচ্ছেদ সম্পর্কে জিজ্ঞাসা করতে পারেন বা তুলনা করতে চাইতে পারেন।"}
        ]
    
    # Display chat history with custom styling
    for idx, message in enumerate(st.session_state.messages):
        if message["role"] != "system":  # Don't display system messages
            message_class = "chat-message user" if message["role"] == "user" else "chat-message assistant"
            with st.container():
                st.markdown(f"<div class='{message_class}'>", unsafe_allow_html=True)
                if message["role"] == "user":
                    st.markdown(f"👤 {message['content']}")
                else:
                    st.markdown(f"🤖 {message['content']}")
                st.markdown("</div>", unsafe_allow_html=True)
    
    # Display sidebar with instructions
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/f/f9/Flag_of_Bangladesh.svg", width=100)
        st.header("📋 ব্যবহার নির্দেশিকা")
        st.markdown("""
        **🔍 অনুসন্ধান করতে:**
        - "বাংলাদেশের সংবিধানের ১লা অনুচ্ছেদ খুঁজুন"
        - "ফিনল্যান্ডের সংবিধানে নাগরিকত্ব সম্পর্কে অনুচ্ছেদ"
        
        **⚖️ তুলনা করতে:**
        - "বাংলাদেশ ও সুইডেনের সংবিধানে মৌলিক অধিকারের তুলনা করুন"
        - "তিনটি দেশের সংবিধানে ধর্মীয় স্বাধীনতার তুলনা"
        
        **❓ সাধারণ প্রশ্ন:**
        - "সংবিধান কি?"
        - "বাংলাদেশের সংবিধান কবে প্রণীত হয়?"
        """)
        
        # Add country selector
        st.subheader("🌐 দেশ নির্বাচন করুন")
        country_options = list(COUNTRIES.keys())
        selected_country = st.selectbox("", country_options)
        
        if st.button("🔎 নির্বাচিত দেশের সংবিধান সম্পর্কে জানুন"):
            # Generate query about selected country's constitution
            query = f"{selected_country}ের সংবিধান সম্পর্কে বিস্তারিত জানতে চাই"
            st.session_state.messages.append({"role": "user", "content": query})
            # Force rerun to process the query
            st.experimental_rerun()
    
    # User input
    if prompt := st.chat_input("আপনার প্রশ্ন লিখুন..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.container():
            st.markdown("<div class='chat-message user'>", unsafe_allow_html=True)
            st.markdown(f"👤 {prompt}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Show translation in progress indicator
        with st.spinner("⏳ প্রসেস করা হচ্ছে..."):
            # Translate user input from Bangla to English for searching
            translated_query = translate_to_english(prompt)
            logger.info(f"Translated query: {translated_query}")
            
            response_text = ""
            
            # Handle different query types
            if is_comparison_query(prompt):
                country = extract_country(prompt)
                articles = search_constitution(translated_query, country_filter=country)
                
                if not articles:
                    response_text = "❌ দুঃখিত, আপনার অনুসন্ধানের সাথে মিলে যাওয়া কোনো অনুচ্ছেদ পাওয়া যায়নি।"
                else:
                    main_article = format_article(articles[0])
                    similar_articles = "\n\n".join(format_article(art) for art in articles[1:])
                    
                    # Generate comparison directly in Bengali using the structured format
                    response_text = generate_constitution_comparison(main_article, similar_articles)
            
            elif is_search_query(prompt):
                country = extract_country(prompt)
                articles = search_constitution(translated_query, country_filter=country)
                
                if not articles:
                    response_text = "❌ দুঃখিত, আপনার অনুসন্ধানের সাথে মিলে যাওয়া কোনো অনুচ্ছেদ পাওয়া যায়নি।"
                else:
                    # Prepare a combined string of all articles in English to send for translation.
 
                    
                    # Create a prompt for the LLM to translate the combined result to Bengali.
                    translation_prompt = f"নীচের ইংরেজি পাঠ্যটি বাংলায় অনুবাদ করুন এবং প্রয়োজনীয় ফরম্যাটিং বজায় রাখুন:\n\n{articles}"
                    messages = [{"role": "user", "content": f"{prompt}\n answer according to following articles {translation_prompt}"}]
                    translated_response = make_openrouter_request(messages)
                    
                    response_text = "📑 **আপনার অনুসন্ধানের ফলাফল:**\n\n" + translated_response


            
            else:
                # General conversation - DIRECTLY in Bengali
                # Get the last 5 messages for context but exclude system messages
                relevant_messages = [msg for msg in st.session_state.messages if msg['role'] != "system"][-5:]
                conversation_history = "\n".join(
                    f"{msg['role']}: {msg['content']}" 
                    for msg in relevant_messages[:-1]  # Exclude the latest user message
                )
                
                # Pass the original Bengali query (not the translated one)
                response_text = generate_general_response(conversation_history, prompt)
            
            # Display response with custom styling
            with st.container():
                st.markdown("<div class='chat-message assistant'>", unsafe_allow_html=True)
                st.markdown(f"🤖 {response_text}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Save to session state
            st.session_state.messages.append({"role": "assistant", "content": response_text})

if __name__ == "__main__":
    main()