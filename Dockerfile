# FractalSemantics Docker Image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Install FractalSemantics in development mode
RUN pip install -e .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash fractalsemantics
USER fractalsemantics

# Set environment
ENV FRACTALSEMANTICS_ENV=production

# Default command
CMD ["python", "-m", "fractalsemantics.fractalsemantics_experiments"]
