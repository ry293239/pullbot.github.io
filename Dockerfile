FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY render/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy app
COPY render/ ./render/
COPY models/ ./models/
COPY data/ ./data/

# Start
CMD ["python", "render/app.py"]
