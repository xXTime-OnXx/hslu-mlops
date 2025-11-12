FROM python:3.11-slim

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .

# make python output visible in real-time
ENV PYTHONUNBUFFERED=1

# Default command with best hyperparameters (default)
CMD ["python", "train.py", \
     "--learning_rate", "1e-4", \
     "--adam_beta1", "0.85", \
     "--warmup_steps", "10", \
     "--lr_schedule", "linear", \
     "--weight_decay", "0.0", \
     "--optimizer", "adamw", \
     "--batch_size", "256"]