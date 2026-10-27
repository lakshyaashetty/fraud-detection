# Start from a small official Linux image with Python 3.14
FROM python:3.14-slim

# XGBoost needs OpenMP on Linux (the same reason you ran brew install libomp on the Mac)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install packages first, so Docker can reuse this step when only code changes
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy in the code and the trained model
COPY src/ src/
COPY app/ app/
COPY models/ models/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
