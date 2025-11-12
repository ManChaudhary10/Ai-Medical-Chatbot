FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app
COPY . .

# Streamlit config
ENV PORT=8080
ENV STREAMLIT_SERVER_PORT=${PORT}
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLECORS=false
ENV STREAMLIT_SERVER_ENABLEWEBSOCKETCOMPRESSION=false
ENV STREAMLIT_SERVER_ENABLESTATICSERVING=true

EXPOSE ${PORT}

CMD ["streamlit", "run", "medibot.py", "--server.port=8080", "--server.address=0.0.0.0"]
