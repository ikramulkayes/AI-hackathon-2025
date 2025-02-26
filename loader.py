import pandas as pd
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

# Load IMDb dataset
df = pd.read_csv('IMDB_movies.csv')

# Initialize Qdrant client
client = QdrantClient("localhost", port=6333)

# Initialize sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create a new collection
client.recreate_collection(
    collection_name="movies",
    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
)

# Process and upload movies
batch_size = 100
for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i+batch_size]
    
    # Generate embeddings for movie titles
    titles = batch['Title'].tolist()
    embeddings = model.encode(titles)
    
    # Prepare points for Qdrant
    points = [
        models.PointStruct(
            id=row['Rank'],
            vector=embedding.tolist(),
            payload={
                'Rank': row['Rank'],
                'Title': row['Title'],
                'Genre': row['Genre'],
                'Description': row['Description']
            }
        )
        for embedding, (_, row) in zip(embeddings, batch.iterrows())
    ]
    
    # Upload points to Qdrant
    client.upsert(
        collection_name="movies",
        points=points
    )
    
    print(f"Uploaded {i+len(batch)} movies")

print("Upload complete!")