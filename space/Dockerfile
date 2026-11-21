FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH CODEOGRAPH_HOME=/home/user/.codeograph \
    BACKEND_URL=http://127.0.0.1:8000 PYTHONUNBUFFERED=1
WORKDIR /home/user/app
# the app straight from GitHub, so a factory rebuild of this Space picks up the latest main
RUN git clone --depth 1 https://github.com/dhruvg0ya1/CodeoGraph src && pip install --no-cache-dir --user "./src[ui]"
RUN codeograph index https://github.com/psf/requests && codeograph index https://github.com/dhruvg0ya1/CodeoGraph
EXPOSE 7860
CMD ["sh", "-c", "codeograph serve --host 127.0.0.1 --port 8000 & exec streamlit run src/ui/app.py --server.port 7860 --server.address 0.0.0.0 --server.enableXsrfProtection false --server.enableCORS false --browser.gatherUsageStats false"]
