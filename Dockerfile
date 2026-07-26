FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml .
RUN mkdir -p backend/app && touch backend/app/__init__.py && pip install --no-cache-dir .
COPY backend backend
RUN pip install --no-cache-dir --no-deps .
ENV PYTHONPATH=/app/backend
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
