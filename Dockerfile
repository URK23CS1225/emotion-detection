# Use a slim CPU-only base image
FROM python:3.9-slim

# Prevent Python from writing .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy requirement file and install only CPU dependencies
COPY requirements.txt .

# Avoid CUDA packages and reduce space usage
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.2.2+cpu torchvision==0.17.2+cpu torchaudio==2.2.2+cpu \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy all source code
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# ✅ Run Streamlit app instead of python app.py
CMD ["streamlit", "run", "Streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
