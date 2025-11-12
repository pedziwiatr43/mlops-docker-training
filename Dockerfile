FROM python:3.10-slim
WORKDIR /app

# Installiere system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "main.py"]
CMD ["--lr", "3e-5", "--batch_size", "32", "--epochs", "3", "--checkpoint_dir", "models/"]
