# Use the official Python 3.9 slim base image for Linux/amd64
FROM python:3.9-slim

# Set the working directory for the application
ENV APP_HOME /app
WORKDIR $APP_HOME

# Copy your application files
COPY . .

# Install other Python dependencies
RUN pip install -r requirements.txt

# Specify the command to run your Python application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]