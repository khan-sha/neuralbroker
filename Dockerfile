FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ src/
COPY config.yaml config.yaml
COPY config.yaml.example config.yaml.example

# Create leads file directory
RUN touch leads.jsonl

EXPOSE 8000

ENV CONFIG_PATH=config.yaml

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
