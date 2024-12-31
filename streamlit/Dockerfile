FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y postgresql-client
RUN pip install --upgrade pip

COPY config/config.yaml /app/config/config.yaml
COPY streamlit/* .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501
