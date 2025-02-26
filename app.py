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

# Load environment variables from .env file
load_dotenv()

# Access the API key from the .env file
api_key = os.getenv("API_KEY")

# Qdrant configuration
QDRANT_URL = os.getenv("QDRANT_URL", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

# Configure the generative AI client
genai.configure(api_key=api_key)

# Initialize the LangChain chat model (using Gemini)
chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=api_key)

# Initialize ConversationBufferMemory
memory = ConversationBufferMemory(return_messages=True)

# Initialize Qdrant client for the "constitutions" collection
qdrant_client = QdrantClient(host=QDRANT_URL, port=QDRANT_PORT)

# Initialize SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Prompt template for constitutional article comparison
constitution_prompt = PromptTemplate(
    input_variables=["article_info", "similar_articles"],
    template="""
You are an expert in constitutional law. Based on the provided article section and similar constitutional articles from other countries, please compare and analyze the content. Provide the similar articles number and section number. Based on country, do the comparison.

Article Section:
{article_info}

Similar Articles:
{similar_articles}

Provide a detailed comparison including any similarities or differences in the rights, responsibilities, or structures described. Also mention the similar articles numbers and section numbers!
"""
)

# LLM chain for constitutional article comparison
constitution_chain = LLMChain(
    llm=chat_model,
    prompt=constitution_prompt
)

# For general conversation (non-comparison/search) we create a simple prompt chain.
general_prompt = PromptTemplate(
    input_variables=["conversation_history", "latest_input"],
    template="""
You are a helpful assistant specializing in constitutional law. Based on the conversation history below and the user's latest input, generate a thoughtful response.

Conversation History:
{conversation_history}

User's Latest Input:
{latest_input}

Response:
"""
)
general_chain = LLMChain(
    llm=chat_model,
    prompt=general_prompt
)

def search_constitution(query_text, country_filter=None):
    """
    Given a query (which could be a specific article section or keyword),
    generate an embedding and search the "constitutions" collection in Qdrant.
    Optionally, filter by a specific country.
    """
    query_embedding = model.encode(query_text)
    
    query_filter = None
    if country_filter:
        query_filter = {
            "must": [
                {"key": "country", "match": {"value": country_filter}}
            ]
        }
    
    search_results = qdrant_client.search(
        collection_name="constitutions",
        query_vector=query_embedding.tolist(),
        query_filter=query_filter,
        limit=3
    )
    print(search_results)
    
    if search_results:
        return [result.payload for result in search_results if result.score >= 0.4]
    return []

def is_comparison_query(user_input):
    """
    Returns True if the user input contains 'compare'
    """
    return "compare" in user_input.lower()

def is_search_query(user_input):
    """
    Returns True if the user input contains 'search' (or similar) along with a country name.
    """
    search_keywords = ['search', 'first', 'law', 'article']
    if any(word in user_input.lower() for word in search_keywords):
        # Check for country presence
        if extract_country(user_input):
            return True
    return False

# Example list of countries to check against
COUNTRIES = ["Bangladesh", "Sweden", "Finland"]

def extract_country(user_input):
    """
    Checks if any known country appears in the user input.
    Returns the first matching country or None.
    """
    for country in COUNTRIES:
        if re.search(rf'\b{country}\b', user_input, re.IGNORECASE):
            return country
    return None

# Streamlit app layout
st.title("Constitutional Article Comparison Chatbot")

# Initialize session state to store chat history with a system message
if "messages" not in st.session_state:
    st.session_state.messages = []
    # System message for context
    system_msg = {"role": "system", "content": "You are a helpful assistant specializing in constitutional law."}
    st.session_state.messages.append(system_msg)
    
    initial_greeting = ("Hello! I'm your constitutional law assistant. You can ask me about constitutional articles, "
                        "compare laws, or ask any general question related to constitutional law.")
    st.session_state.messages.append({"role": "assistant", "content": initial_greeting})

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Enter your constitutional law query..."):
    # Append the user prompt to session state as a dict
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Initialize a variable for response_text
    response_text = ""
    
    # Decide on action based on input
    if is_comparison_query(prompt):
        # Comparison query: use vector search and comparison chain.
        country = extract_country(prompt)
        if country:
            articles = search_constitution(prompt, country_filter=country)
        else:
            articles = search_constitution(prompt)
        
        if articles:
            main_article = articles[0]['text']
            similar_articles = "\n\n".join(
                f"Country: {art.get('country', 'Unknown')}\nArticle: {art.get('text', '')}"
                for art in articles[1:]
            )
        else:
            main_article = "No matching constitutional article found."
            similar_articles = "No similar articles found."
        
        # Generate comparison using the LLM chain
        response_data = constitution_chain.invoke({
            "article_info": main_article,
            "similar_articles": similar_articles
        })
        response_text = response_data.get('text', "Sorry, I couldn't generate a comparison.")
        
    elif is_search_query(prompt):
        # Search query: Look for articles in the vector database (e.g., "search first 3 laws in Bangladesh")
        country = extract_country(prompt)
        articles = search_constitution(prompt, country_filter=country) if country else search_constitution(prompt)
        
        if articles:
            # For search queries, simply return the articles.
            response_text = "\n\n".join(
                f"Country: {art.get('country', 'Unknown')}\nArticle: {art.get('text', '')}"
                for art in articles
            )
        else:
            response_text = "No matching constitutional articles found."
    else:
        # Otherwise, respond based on conversation history.
        # Build a conversation history string from the messages (excluding the system message).
        conversation_history = "\n".join(
            f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages if msg['role'] != "system"
        )
        # Use the general chain to generate a response
        response_data = general_chain.invoke({
            "conversation_history": conversation_history,
            "latest_input": prompt
        })
        response_text = response_data.get('text', "Sorry, I didn't understand that.")
    
    # Display assistant's response
    with st.chat_message("assistant"):
        st.markdown(response_text)
    
    # Append the assistant response to session state
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    
    # Update conversation memory (if needed)
    memory.chat_memory.add_user_message(prompt)
    memory.chat_memory.add_ai_message(response_text)
