from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from sqlalchemy import create_engine, text
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import yaml
import uvicorn
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
    config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False},
)

ner_model = config["ner_model"]["ner_name"]
ner_tokenizer = config["ner_model"]["tokenizer"]

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
        sql = text(
            f"""
            SELECT id, content, embedding <=> (:query_embedding)::vector AS similarity
            FROM documents
            ORDER BY similarity
            LIMIT :top_k;
        """
        )

        # Execute the query
        with engine.connect() as connection:
            results = connection.execute(
                sql,
                {
                    "query_embedding": query_embedding.tolist(),
                    "top_k": config["retrieval"]["top_k"],
                },
            ).fetchall()

        # Format the results
        formatted_results_start = [{"content": row[1]} for row in results]
        return formatted_results_start

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/search_ner")
async def search_ner(chunks: Query):
    """
    Perform NER in query 1.
    """
    try:
        ner_pipeline = pipeline(
            "ner",
            model=ner_model,  # Specifying the model
            tokenizer=ner_tokenizer,  # Specifying the tokenizer
        )

        result = ner_pipeline(
            chunks.query
        )  # using result from all chunks to find disease names

        diseases = []
        for entity in result:
            if entity["entity"] == "Disease":
                diseases.append(entity["word"])
            elif entity["entity"] == "Disease Continuation" and diseases:
                diseases[-1] += f" {entity['word']}"

        #        diseases = ', '.join(diseases)

        for i in range(len(diseases)):
            diseases[i] = diseases[i].replace(" ##", "")

        dis = ", ".join(diseases)
        return dis
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}, Input: {chunks.query}",
        )


@app.post("/similarity_search_icd")
async def similarity_search_icd(dis: Query):
    """
    Perform similarity search in ICD-11
    """
    try:
        db_name = "/app/faiss_db"

        # loading previously saved faiss index to compute similarity search
        vector_icd = FAISS.load_local(
            db_name, embeddings_model, allow_dangerous_deserialization=True
        )

        # Search for needed ICD-11 Code
        query_embedding = embeddings_model.encode(dis.query)

        res = vector_icd.similarity_search_with_score_by_vector(query_embedding, k=1)
        results_icd = [r for r, _ in res]

        # Format the results
        formatted_results_end = [
            {"code": row.metadata["code"], "disease": row.page_content}
            for row in results_icd
        ]
        return formatted_results_end

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
