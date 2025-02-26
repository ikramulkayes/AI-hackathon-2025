import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableSequence
from langchain_core.exceptions import OutputParserException

# Load environment variables from .env file
load_dotenv()

# Access the API key from the .env file
api_key = os.getenv("API_KEY")

# Configure the generative AI client
genai.configure(api_key=api_key)

# Initialize the LangChain chat model
chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=api_key)

# Initialize ConversationBufferMemory
memory = ConversationBufferMemory(return_messages=True)

# Prompt to handle career expert responses
career_expert_prompt_template = PromptTemplate(
    input_variables=["text"],
    template="""
You are an experienced and knowledgeable career expert assistant. Provide professional, thoughtful advice on the following career-related query:

Query: "{text}"

Your response should be:
- Professional
- Informative
- Actionable

Provide your advice below:
"""
)

# Translation from Bengali to English prompt
translation_prompt_template = PromptTemplate(
    input_variables=["text"],
    template="""Translate the following text to English and return it in the following JSON format: 
    {{
        "text": "<translated_text>"
    }}

    Text to translate: "{text}"
    """
)

# Translation from English to Bengali prompt
translate_to_bengali_prompt = PromptTemplate(
    input_variables=["text"],
    template="""Translate the following English text to Bengali and return it in the following JSON format: 
    {{
        "text": "<translated_text>"
    }}

    Text to translate: "{text}"
    """
)

# LLM chains
# Define the chain for translation from Bengali to English
translate_to_english_chain = LLMChain(
    llm=chat_model,
    prompt=translation_prompt_template
)

# Define the chain for generating a career response
career_response_chain = LLMChain(
    llm=chat_model,
    prompt=career_expert_prompt_template
)

# Define the chain for translating English to Bengali
translate_to_bengali_chain = LLMChain(
    llm=chat_model,
    prompt=translate_to_bengali_prompt
)

# Chain the processes
chain = (
    {"text": RunnablePassthrough()}  # Input passthrough
    | translate_to_english_chain     # Translate Bengali to English
    | career_response_chain          # Generate the career advice
    | translate_to_bengali_chain     # Translate the response back to Bengali
)

# Streamlit app layout
st.title("Career Expert Assistant (Bengali Input Supported)")

# Initialize session state to store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    initial_greeting = "হ্যালো! আমি আপনার ক্যারিয়ার বিশেষজ্ঞ সহকারী। কিভাবে আমি আপনাকে সাহায্য করতে পারি?"
    st.session_state.messages.append({"role": "assistant", "content": initial_greeting})

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("আপনার ক্যারিয়ার সম্পর্কিত প্রশ্ন লিখুন..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response using the chain, passing the user's prompt
    try:
        response = chain.invoke({"text": prompt})  # Pass the prompt
        print(response)
        bengali_response = response["text"]  # Extract the translated response in Bengali
        
        # Display bot response in chat message container
        with st.chat_message("assistant"):
            st.markdown(bengali_response)
        
        # Add bot response to chat history
        st.session_state.messages.append({"role": "assistant", "content": bengali_response})

    except OutputParserException as e:
        st.error(f"Error parsing response: {str(e)}")

    # Update memory
    memory.chat_memory.add_user_message(prompt)
    memory.chat_memory.add_ai_message(bengali_response)