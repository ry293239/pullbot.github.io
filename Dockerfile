FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

COPY render/requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY render/ ./render/
COPY models/pullbot.gguf ./models/pullbot.gguf
COPY data/wordbank.json ./data/wordbank.json

EXPOSE 7860

CMD ["python", "render/app.py"]
