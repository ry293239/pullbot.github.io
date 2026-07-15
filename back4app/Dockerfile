FROM python:3.10-slim

WORKDIR /app

COPY back4app/requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY back4app/ .

EXPOSE 7860

CMD ["python", "app.py"]
