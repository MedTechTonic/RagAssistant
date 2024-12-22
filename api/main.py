import yaml
import httpx
import logging
import models, schemas
from fastapi import FastAPI
from contextlib import asynccontextmanager
from database import engine
from utils import initialize_database, initialize_llm_client, retrieve_embeddings
from sqlalchemy.sql import text
import os

logger = logging.getLogger(__name__)
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
llm = initialize_llm_client(config)

@app.post("/query/")
async def query(query: str):
    try:
        context_json = await retrieve_embeddings(query, config)
        context = '\n'.join([content["content"] for content in context_json])
        response = llm.chat.completions.create(
                model=config["llm"]["model"],
                messages=[
                    {"role": "system", "content": config["llm"]["system_prompt"]},
                    {"role": "user", "content": f"{context}\n{query}"},
                ],
                temperature=config["llm"]["temperature"],
                stream=True,
            )

        generated_response = ""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                generated_response += chunk.choices[0].delta.content
        return {"context": context, "response": generated_response}
    except Exception as e:
        logger.error(f"An error occurred while processing the query: {e}")
        return {"error": "An error occurred while processing your request."}