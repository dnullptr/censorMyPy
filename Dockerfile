FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# התקנת חבילות מערכת (System Dependencies)
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-venv \
    ffmpeg \
    git \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/bin/python
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# התקנת PyTorch בצורה מבודדת (מונע קונפליקטים)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

COPY requirements_clean.txt .

# כאן הקסם: אנחנו אומרים ל-pip לא להשתגע עם תלויות
RUN pip install --no-cache-dir -r requirements_clean.txt

ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib/python3.10/dist-packages/tensorrt_libs"
COPY . .

# אין צורך ב-EXPOSE אם אתה לא מריץ שרת
# EXPOSE 8000 

CMD ["sleep", "infinity"]