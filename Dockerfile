# Use the standard Python 3.10 image 
FROM python:3.10 

# Set the working directory inside the container to /app
WORKDIR /app

# --- NEW FIX: Install Rust Compiler for 'river' ---
# Rust is required to build certain components of the river library from source.
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/root/.cargo/bin:$PATH"

# --- GUARANTEE CPU-ONLY INSTALLATION ---
ENV TORCH_INSTALL_VIA_PIP=True
ENV FORCE_CUDA=0

# 1. Copy requirements
COPY requirements.txt .

# Install dependencies. We keep the timeout for network stability.
RUN pip install --no-cache-dir --timeout 600 -r requirements.txt

# 2. Now copy the actual code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Ensure the app root is in the Python path
ENV PYTHONPATH="/app"

# 3. Define the entry point
CMD ["python", "scripts/run_incremental_batch.py"]