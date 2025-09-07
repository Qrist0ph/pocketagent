# Python 3.11 image with pip
FROM mcr.microsoft.com/devcontainers/python:3.11

# Set working directory
WORKDIR /workspace/pocketrag

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Default command
CMD ["python", "rag.py"]
