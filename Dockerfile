# LiveKit agent worker
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml .
COPY agent agent
COPY crm crm
COPY evals evals
COPY demo demo
# ".[demo]" adds the web server for the demo website
RUN pip install --no-cache-dir ".[demo]"
# bake VAD + turn-detector weights into the image
RUN python -m agent.main download-files
# the code version saved in every call record (docker compose build --build-arg GIT_SHA=$(git rev-parse --short HEAD))
ARG GIT_SHA=""
ENV GIT_SHA=${GIT_SHA}
CMD ["python", "-m", "agent.main", "start"]
