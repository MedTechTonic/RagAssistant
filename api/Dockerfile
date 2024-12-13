FROM python:3.9-slim

RUN apt-get update && apt-get install -y postgresql-client
WORKDIR /app

COPY api/* /app
COPY config/config.yaml /app/config/config.yaml

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "main.py"]