import os
import re
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

# Initialize sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create a new collection for constitutions (adjust vector size if needed)
client.recreate_collection(
    collection_name="constitutions",
    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
)
def normalize_text(text):
    # Replace newline characters with a space
    return re.sub(r'\s*\n\s*', ' ', text).strip()


def extract_text_from_pdf(pdf_path):
    """
    Try to extract text using PyMuPDF.
    If text is insufficient, fall back to OCR.
    """
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join(page.get_text() for page in doc)
        # If extracted text is too short, try OCR (assume scanned image)
        if len(text.strip()) < 100:
            raise ValueError("Text too short, possibly a scanned PDF")
    except Exception as e:
        print(f"Falling back to OCR for {pdf_path}: {e}")
        images = convert_from_path(pdf_path)
        text = "\n".join(pytesseract.image_to_string(img) for img in images)
    return text

# Define a text splitter to break down large documents into manageable chunks.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# Process each PDF in the folder
pdf_files = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))
batch_size = 10  # adjust as needed for your processing and memory

all_points = []  # will accumulate all points for upserting
id_counter = 0   # initialize a global counter outside the loop

for pdf_file in pdf_files:
    country_name = os.path.splitext(os.path.basename(pdf_file))[0]
    print(f"Processing constitution for: {country_name}")
    text = extract_text_from_pdf(pdf_file)
    
    # Split text into chunks
    chunks = text_splitter.split_text(text)
    
    # Generate embeddings for each chunk
    embeddings = model.encode(chunks, show_progress_bar=True)
    
    # Prepare points for Qdrant using an unsigned integer as ID.
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        point = models.PointStruct(
            id=id_counter,
            vector=embedding.tolist(),
            payload={
                "country": country_name,
                "chunk_index": i,
                "text": normalize_text(chunk)
            }
        )
        all_points.append(point)
        id_counter += 1
        
    # Optional: Upsert in batches to avoid sending too many points at once
    if len(all_points) >= batch_size:
        client.upsert(collection_name="constitutions", points=all_points)
        print(f"Upserted {len(all_points)} points.")
        all_points = []

# Upsert any remaining points
if all_points:
    client.upsert(collection_name="constitutions", points=all_points)
    print(f"Upserted final {len(all_points)} points.")

print("Upload complete!")
