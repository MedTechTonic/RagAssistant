from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
import yaml
import numpy as np
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load configuration settings from YAML file
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize FastAPI app
app = FastAPI()

# Initialize SentenceTransformer
embeddings_model = SentenceTransformer(
    config["embedding_model"]["model_name"],
    trust_remote_code=True,
    device="cpu",
    config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False}
)

# Database connection
DATABASE_URL = f"postgresql://{config['database']['user']}:{config['database']['password']}@db:{config['database']['port']}/{config['database']['name']}"
engine = create_engine(DATABASE_URL)

# Input model for the query
class Query(BaseModel):
    query: str

@app.post("/similarity_search")
async def similarity_search(query: Query):
    """
    Perform similarity search using SQLAlchemy and a custom cosine similarity query.
    """
    try:
        # Encode the query to generate its embedding
        query_embedding = embeddings_model.encode([query.query])[0]
        

        # Build a SQL query to compute cosine similarity
        sql = text(f"""
        SELECT id, content, embedding <=> (:query_embedding)::vector AS similarity
        FROM documents
        ORDER BY similarity
        LIMIT :top_k;

        """)

        # Execute the query
        with engine.connect() as connection:
            results = connection.execute(
                sql,
                {
                    "query_embedding": query_embedding.tolist(),
                    "top_k": config["retrieval"]["top_k"],
                }
            ).fetchall()

        # Format the results
        formatted_results = [
            {"content": row[1]}
            for row in results
        ]
        return formatted_results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
