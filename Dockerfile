FROM python:3.11-slim

# Install system dependencies for OpenCV and YOLO
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Ensure weights are included (assumes they are in app/weights)
# Create logs directory if needed
RUN mkdir -p logs

# Expose the API port
EXPOSE 5000

# Set Python Path and real-time logging
ENV PYTHONPATH=.
ENV PYTHONUNBUFFERED=1

# Use gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "600", "--workers", "1", "app.main:app"]
