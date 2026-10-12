# LiveKit agent worker
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml .
COPY agent agent
COPY crm crm
COPY evals evals
COPY loadtest loadtest
RUN pip install --no-cache-dir .
# bake VAD + turn-detector weights into the image
RUN python -m agent.main download-files
CMD ["python", "-m", "agent.main", "start"]
