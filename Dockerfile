FROM python:3.12-slim
# git: bpypolyskel installs from GitHub. gcc/python3-dev: mathutils ships no Linux wheels.
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential python3-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir . uvicorn
ENV PORT=8080 ROOFMEASURE_OUT_DIR=/data/reports MPLBACKEND=Agg
RUN mkdir -p /data/reports
EXPOSE 8080
CMD ["roofmeasure-mcp-http"]
