FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Paksa instalasi versi CPU agar hemat ruang dan memori
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render akan menyuntikkan $PORT secara otomatis saat runtime
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]