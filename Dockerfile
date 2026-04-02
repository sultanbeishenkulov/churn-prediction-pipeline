FROM python:3.11-slim

WORKDIR /app

# Install API deps first (cached layer)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Install training deps (only needed at build time)
RUN pip install --no-cache-dir optuna mlflow

# Copy source + data
COPY load.py .
COPY engineer.py .
COPY train.py .
COPY main.py .
COPY data/raw/telco_churn.csv data/raw/telco_churn.csv

# Train model at build time
RUN python train.py

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
