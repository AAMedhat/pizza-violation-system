FROM python:3.10-slim

WORKDIR /app

# System dependencies for OpenCV and PyTorch compatibility
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1-mesa-glx \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy your application code
COPY . .

# Set PYTHONPATH so Python can find internal modules
ENV PYTHONPATH=/app

# Install PyTorch 2.7.0 + CUDA 11.8 explicitly (to match local environment)
RUN pip install --no-cache-dir torch==2.7.0+cu118 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

# Install other Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

# Keeps container alive if no command passed (overridden by docker-compose)
CMD ["tail", "-f", "/dev/null"]


# The actual command will be set via docker-compose.yml,
# so we don't need an ENTRYPOINT or CMD here.
