# Project README
## Overview
This project is a multi-service application that leverages Docker Compose to manage various services, including a PostgreSQL database, a retriever service, an API service, and a Streamlit service. The application is designed to handle queries, retrieve embeddings, perform named entity recognition (NER), and map to ICD-11 codes using a large language model (LLM).

```
git clone <repository-url>
cd <repository-directory>
Build and start the services using Docker Compose:
docker-compose up --build
```


## Usage
Start the services using Docker Compose:

```
docker-compose up --build
```
Access the API service at http://localhost:8000.
Access the Streamlit service at http://localhost:8501.

## env file
Add env file in root with:
```
API_KEY=your_api_key
```
By default it's using mistral-large-latest

## Ragas Metrics
| Metric                | Average Score |
|-----------------------|---------------|
| **Context Precision** | 0.923077      |
| **Faithfulness**      | 0.896722      |
| **Answer Relevancy**  | 0.934879      |
| **Context Recall**    | 0.607143      |
