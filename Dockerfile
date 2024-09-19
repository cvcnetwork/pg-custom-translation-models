# Use the official Python 3.9 slim base image for Linux/amd64
<<<<<<< HEAD
FROM python:3.9-slim AS builder

# Set environment variables
ENV PIP_DEFAULT_TIMEOUT=100
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
=======
FROM python:3.9-slim
>>>>>>> 7aa91347c3122f796bbe58af93c509e425e4baf1

# Set the working directory for the application
ENV APP_HOME /app
WORKDIR $APP_HOME

# Copy the requirements file into the image
<<<<<<< HEAD
COPY requirements.txt .

# Install git and PyTorch dependencies
RUN apt-get update && apt-get install -y git && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other Python dependencies
RUN pip install -r requirements.txt

# Start a new stage
FROM python:3.9-slim

# Set the working directory
ENV APP_HOME /app
WORKDIR $APP_HOME

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy your application files
COPY . .

# Clean up
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Specify the command to run your Python application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
=======
COPY requirements.txt ./

# Install Python dependencies from requirements.txt
RUN apt-get update && apt-get install -y git && \
    pip install -r requirements.txt && \
    rm -rf /var/lib/apt/lists/*

# Copy your Python application files into the image
COPY *.py .
COPY languages.csv .

# Specify the command to run your Python application
CMD ["python", "translate.py"]
>>>>>>> 7aa91347c3122f796bbe58af93c509e425e4baf1
