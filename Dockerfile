# Use the official Python 3.9 slim base image for Linux/amd64
FROM python:3.9-slim

# Set the working directory for the application
ENV APP_HOME /app
WORKDIR $APP_HOME

# Copy the requirements file into the image
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
