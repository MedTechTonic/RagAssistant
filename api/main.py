import yaml
import httpx
import logging
import models, schemas
from fastapi import FastAPI
from langchain.chat_models import ChatOpenAI
from contextlib import asynccontextmanager
from database import engine
from utils import initialize_database
from sqlalchemy.sql import text
import os

# Initialize the logger
logger = logging.getLogger(__name__)
# Load configuration settings from YAML file
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

logger.info("Starting initialization of database...")
models.Base.metadata.create_all(bind=engine)
logger.info("Database initialized.")

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

app = FastAPI(lifespan=lifespan)

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
logging.info(os.getenv("OPENAI_API_KEY"))
llm = ChatOpenAI(model_name=config['llm']['model_name'], temperature=config['llm']['temperature'])

async def fetch_similarity_search(query: str):
    retriever_url = f"http://retriever:8001/similarity_search"
    async with httpx.AsyncClient() as client:
        response = await client.post(retriever_url, json={"query": query})
        response.raise_for_status()
        return response.json()

@app.post("/query/")
async def query(query: str):
    try:
        answer = []
        context = await fetch_similarity_search(query)
        async for chunk in llm.astream(
            prompt=config["llm"]["prompt"],
            messages=[{"role": "user", "content": f"Context:\n{context}\nQuery:\n{query}"}]
        ):
            answer.append(chunk)
        return {"response": answer}
    except Exception as e:
        logger.error(f"An error occurred while processing the query: {e}")
        return {"error": "An error occurred while processing your request."}