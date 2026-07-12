FROM python:3.12-slim

LABEL org.opencontainers.image.title="Triplet Annotator" \
      org.opencontainers.image.description="Streamlit tool for human validation of LLM-extracted triplets"

WORKDIR /app

# Patch OS-level packages in the base image (addresses base-image CVEs), then
# drop apt lists to keep the layer small. Rebuild periodically to stay current.
RUN apt-get update && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

# Only streamlit is needed — the annotation app never calls a model.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code (build context is annotation_app/; see .dockerignore for exclusions).
COPY . .

# Inputs and saves live on a mounted volume at /data, NOT baked into the image,
# so you can update the extraction files without rebuilding. See deploy/k8s.yaml.
#   ANNOTATION_DATA_DIR : holds <name>/<task>_<name>.json extraction files
#   ANNOTATION_SAVE_DIR : per-(task, annotator) saves + snapshots/
# ANNOTATION_PASSWORD is the shared access password — inject it at run time from
# a secret (docker run -e / k8s Secret); never bake it into the image. The app
# fail-closes (won't start) if it is unset.
ENV ANNOTATION_DATA_DIR=/data/extracted_triplets \
    ANNOTATION_SAVE_DIR=/data/saves \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHERUSAGESTATS=false

# Persistent storage for inputs + annotation outputs — mount a real volume here,
# or annotations are lost on restart.
RUN mkdir -p /data
VOLUME ["/data"]

EXPOSE 8501

# Fail the container if Streamlit stops serving (stdlib only; no curl in -slim).
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').status==200 else 1)"

# Example run (provide the password + a persistent volume):
#   docker run -p 8501:8501 -e ANNOTATION_PASSWORD='...' -v mobiko-data:/data <IMAGE>
#
# 0.0.0.0 so the cluster Service can reach it. Single process serves all annotators.
CMD ["streamlit", "run", "streamlit_app_triplets.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
