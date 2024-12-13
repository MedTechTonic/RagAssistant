# retriever/Dockerfile
FROM python:3.9-slim

RUN apt-get update && apt-get install -y postgresql-client
RUN pip install --upgrade pip
WORKDIR /app

COPY retriever/* /app
COPY config/config.yaml /app/config/config.yaml

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8001

CMD ["uvicorn", "retriever:app", "--host", "0.0.0.0", "--port", "8001"]
