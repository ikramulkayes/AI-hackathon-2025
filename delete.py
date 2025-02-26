

import os
import glob
import fitz  # PyMuPDF for extracting text from PDF
import pytesseract
from pdf2image import convert_from_path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Folder containing PDF constitutions
PDF_FOLDER = "constitutions_folder"  # update with your folder path

# Initialize Qdrant client (ensure Qdrant is running locally)
client = QdrantClient("localhost", port=6333)







client.delete_collection(collection_name="constitutions")
print("Collection 'constitutions' deleted successfully.")
