FROM python:3.11-slim

WORKDIR /app

# Instal library sistem Linux dasar yang dibutuhkan oleh OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Paksa hapus OpenCV versi GUI bawaan YOLO dan pasang versi headless
RUN pip uninstall -y opencv-python && pip install --no-cache-dir opencv-python-headless

COPY . .

# Render akan menyuntikkan $PORT secara otomatis saat runtime
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]