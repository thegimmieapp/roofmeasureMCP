FROM python:3.12-slim
# git: bpypolyskel (pure Python) installs from GitHub. No compiler needed:
# roofmeasure ships its own pure-Python mathutils subset (src/roofmeasure/_mathutils.py).
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir . uvicorn
ENV PORT=8080 ROOFMEASURE_OUT_DIR=/data/reports MPLBACKEND=Agg
RUN mkdir -p /data/reports
EXPOSE 8080
CMD ["roofmeasure-mcp-http"]
