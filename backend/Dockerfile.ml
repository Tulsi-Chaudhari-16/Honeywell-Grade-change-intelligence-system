FROM python:3.12-slim

WORKDIR /app

# We need to copy requirements.txt and requirements-ml.txt if any.
# In the original structure, requirements-ml.txt is at c:\Desktop\Honeywells\requirements-ml.txt
# and this dockerfile will be at c:\Desktop\Honeywells\backend\Dockerfile.ml
# So context is c:\Desktop\Honeywells
COPY requirements-ml.txt .
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-ml.txt
RUN pip install fastapi uvicorn

COPY backend/app/ml ./app/ml
COPY backend/app/__init__.py ./app/__init__.py
COPY backend/ml_api.py .

CMD ["python", "ml_api.py"]
