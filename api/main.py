import yaml
import httpx
import logging
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from sqlalchemy.sql import text
from database import engine
from utils import initialize_database, initialize_llm_client, retrieve_embeddings
from schemas import QueryPayload
import models

# Initialize logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load configuration
try:
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError as e:
    logger.critical(f"Configuration file not found: {e}")
    raise
except yaml.YAMLError as e:
    logger.critical(f"Error parsing configuration file: {e}")
    raise

# Initialize database
logger.info("Starting initialization of database...")
try:
    models.Base.metadata.create_all(bind=engine)
    logger.info("Database initialized.")
except Exception as e:
    logger.error(f"Database initialization failed: {e}")
    raise

# Define lifespan for FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Starting data loading process.")
        initialize_database()  # Pass config to load_data
        logger.info("Data loading completed.")
    except Exception as e:
        logger.warning(f"Service starts without any data in the DB caused by: {e}")

    yield
    logger.info("App shutting down")

# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)

# Initialize LLM client
try:
    llm = initialize_llm_client(config)
except Exception as e:
    logger.critical(f"Failed to initialize LLM client: {e}")
    raise

@app.post("/query/")
async def query(payload: QueryPayload):
    query_text = payload.query
    if not query_text:
        raise HTTPException(status_code=400, detail="Query parameter is required.")

    try:
        # Retrieve embeddings based on query
        context_json = await retrieve_embeddings(query_text, config)
        context = '\n'.join([content["content"] for content in context_json])

        # Generate response from LLM
        response = llm.chat.completions.create(
            model=config["llm"]["model"],
            messages=[
                {"role": "system", "content": config["llm"]["system_prompt"]},
                {"role": "user", "content": f"{context}\n{query_text}"},
            ],
            temperature=config["llm"]["temperature"],
            stream=True,
        )

        # Aggregate streamed response
        generated_response = ""
        for chunk in response:
            delta_content = chunk.choices[0].delta.content
            if delta_content:
                generated_response += delta_content

        return {"context": context, "response": generated_response}

    except Exception as e:
        logger.error(f"An error occurred while processing the query: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your request.")
