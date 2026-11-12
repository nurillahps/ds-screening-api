FROM python:3.11-slim

WORKDIR /app

# Instal library sistem Linux
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Instal dependensi dasar (tanpa opencv)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Hapus opencv-python jika ada (baikan error jika tidak ada) lalu instal versi headless
RUN pip uninstall -y opencv-python || true
RUN pip install --no-cache-dir opencv-python-headless==4.10.0.84

COPY . .

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]