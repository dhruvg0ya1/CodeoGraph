# CodeoGraph: the REST API on :8000 and the Streamlit UI on :8501 in one image.
#   docker build -t codeograph . && docker run -p 8000:8000 -p 8501:8501 -v codeograph:/data codeograph
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY codeograph ./codeograph
COPY ui ./ui
RUN pip install --no-cache-dir ".[ui,gemini]"
ENV CODEOGRAPH_HOME=/data BACKEND_URL=http://localhost:8000
VOLUME /data
EXPOSE 8000 8501
CMD ["sh", "-c", "codeograph serve --host 0.0.0.0 --port 8000 & exec streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0"]
