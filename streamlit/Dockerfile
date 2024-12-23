FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y postgresql-client
RUN pip install --upgrade pip

COPY streamlit/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY streamlit/app.py .

EXPOSE 8501