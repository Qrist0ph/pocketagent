# Python 3.11 image with pip
FROM mcr.microsoft.com/devcontainers/python:3.11

# Set working directory
WORKDIR /workspace/pocketrag

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Add src directory to Python path
ENV PYTHONPATH="/workspace/pocketrag/src"

# Expose port 8000 for the FastAPI app
EXPOSE 8000

# Default command to run the FastAPI app with uvicorn
CMD ["uvicorn", "src.api.index:app", "--reload", "--host", "0.0.0.0"]
