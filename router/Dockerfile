FROM python:3.11-slim
WORKDIR /app
COPY router/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY router/ .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
