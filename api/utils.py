import numpy as np
import os
import httpx
import pandas as pd
from openai import OpenAI
from sqlalchemy import select
from tqdm import tqdm
from database import SessionLocal
from models import Document, ICDDocument
import logging

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def initialize_llm_client(config: dict):
    """
    Initialize a client for interacting with the LLM.

    Args:
        config: A configuration dictionary containing the LLM's base URL and API key.

    Returns:
        An OpenAI client object.

    Raises:
        Exception: If the LLM client could not be initialized.
    """
    try:
        llm_client = OpenAI(
            base_url=config["llm"]["base_url"], api_key=os.getenv("API_KEY")
        )
        logger.info("LLM client initialized successfully.")
        return llm_client
    except Exception as e:
        logger.error(f"Failed to initialize LLM client: {e}")
        raise

async def retrieve_embeddings(query: str, config: dict):
    """
    Asynchronously retrieve embeddings for a given query using a remote retrieval service.

    Args:
        query: A string containing the query for which embeddings are to be retrieved.
        config: A dictionary containing configuration details, including the host and port of the retrieval service.

    Returns:
        A JSON response containing the embeddings for the query.
    """

    retriever_url = f"http://{config['retrieval']['host']}:{config['retrieval']['port']}/similarity_search"
    async with httpx.AsyncClient() as client:
        response = await client.post(retriever_url, json={"query": query}, timeout=1000.0)
        response.raise_for_status()
        return response.json()
    
async def retrieve_ner(chunks: str, config: dict):
    """
    Asynchronously retrieve named entities for a given query using a remote retrieval service.

    Args:
        chunks: A string containing the query for which named entities are to be retrieved.
        config: A dictionary containing configuration details, including the host and port of the retrieval service.

    Returns:
        A JSON response containing the named entities for the query.
    """
    retriever_url = f"http://{config['retrieval']['host']}:{config['retrieval']['port']}/search_ner"
    async with httpx.AsyncClient() as client:
        response = await client.post(retriever_url, json={"query": chunks}, timeout=1000.0)
        response.raise_for_status()
        return response.json()    
     
def insert_embeddings_from_parquet_icd(file_path_parquet: str, file_path_npy: str, batch_size=1000):
    """
    Insert embeddings from Parquet and .npy files into the database.

    Args:
        file_path_parquet (str): The path to the Parquet file containing the embeddings.
        file_path_npy (str): The path to the .npy file containing the embeddings.
        batch_size (int, optional): The number of embeddings to insert in each batch. Defaults to 1000.
    """
    session = SessionLocal()
    try:
        if session.query(ICDDocument.id).first() is not None:
            logger.info("Database already contains embeddings. Skipping insertion.")
            return

        logger.info("Loading data from Parquet and .npy files...")
        df = pd.read_parquet(file_path_parquet)
        embeddings = np.load(file_path_npy)

        if len(df) != len(embeddings):
            raise ValueError("Mismatch between the number of rows in Parquet and embeddings!")

        logger.info(f"Loaded {len(df)} rows and {len(embeddings)} embeddings.")

        for start_idx in tqdm(range(0, len(df), batch_size), desc="Inserting batches"):
            end_idx = start_idx + batch_size
            batch_data = []
            for idx in range(start_idx, end_idx):
                if idx >= len(df):
                    break
                row = df.iloc[idx] 
                embedding = embeddings[idx]
                batch_data.append(ICDDocument(content=row[0], embedding=embedding.tolist()))

            session.bulk_save_objects(batch_data)
            session.commit()
            logger.info(f"Batch {start_idx // batch_size + 1} inserted successfully.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        session.rollback()
    finally:
        session.close()
        logger.info("Session closed.")

async def retrieve_icd(dis: str, config: dict):
    """
    Asynchronously retrieve ICD code for a given disease/disorder using a remote retrieval service.

    Args:
        dis (str): The disease/disorder for which the ICD code is to be retrieved.
        config (dict): A dictionary containing configuration details, including the host and port of the retrieval service.

    Returns:
        A JSON response containing the ICD code for the given disease/disorder.
    """
    retriever_url = f"http://{config['retrieval']['host']}:{config['retrieval']['port']}/similarity_search_icd"
    async with httpx.AsyncClient() as client:
        response = await client.post(retriever_url, json={"query": dis}, timeout=1000.0)
        response.raise_for_status()
        return response.json()
    
def insert_embeddings_from_parquet(file_path_parquet: str, file_path_npy: str, batch_size=1000):
    
    """
    Inserts embeddings from a Parquet file and a .npy file into the database in batches.

    Args:
        file_path_parquet (str): The path to the Parquet file containing the text data.
        file_path_npy (str): The path to the .npy file containing the embeddings.
        batch_size (int, optional): The number of rows to insert in each batch. Defaults to 1000.
    """
    session = SessionLocal()
    try:
        if session.query(Document.id).first() is not None:
            logger.info("Database already contains embeddings. Skipping insertion.")
            return

        # Load data
        logger.info("Loading data from Parquet and .npy files...")
        df = pd.read_parquet(file_path_parquet)
        embeddings = np.load(file_path_npy)

        if len(df) != len(embeddings):
            raise ValueError("Mismatch between the number of rows in Parquet and embeddings!")

        logger.info(f"Loaded {len(df)} rows and {len(embeddings)} embeddings.")

        # Process data in batches
        for start_idx in tqdm(range(0, len(df), batch_size), desc="Inserting batches"):
            end_idx = start_idx + batch_size
            batch_data = []
            for idx in range(start_idx, end_idx):
                if idx >= len(df):
                    break
                row = df.iloc[idx]  # Get the row as a Series
                embedding = embeddings[idx]
                batch_data.append(Document(content=row[0], embedding=embedding.tolist()))

            session.bulk_save_objects(batch_data)
            session.commit()  # Commit the batch
            logger.info(f"Batch {start_idx // batch_size + 1} inserted successfully.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        session.rollback()
    finally:
        session.close()
        logger.info("Session closed.")

def initialize_database():
    """
    Initialize the database by loading embeddings and inserting them.
    """
    try:
        logger.info("Initializing the database...")
        insert_embeddings_from_parquet("data/data.parquet", "data/data.npy", batch_size=100)
        insert_embeddings_from_parquet_icd("data/data_icd11.parquet", "data/data_icd11.npy", batch_size=100)
        logger.info("Database initialization complete.")
    except Exception as e:
        logger.error(f"An error occurred while initializing the database: {e}")
