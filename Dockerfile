# Container image for view_eval_results.py, intended for Cloud Run.
#
# Configuration is via env vars; the Cloud Run service sets at minimum
#   VIEWER_ROOT=gs://bucket/prefix
# and inherits PORT from the Cloud Run runtime.
#
# Auth: GCS reads use Application Default Credentials, which on Cloud Run
# resolve to the runtime service account passed at deploy time
# (`gcloud run deploy --service-account ...`). Grant that SA roles/storage
# .objectViewer on the bucket.
#
# Build:    gcloud builds submit --tag gcr.io/PROJECT/image2prompt-viewer
# Or local: docker build -t image2prompt-viewer .

FROM python:3.12-slim AS runtime

# Keep the layer minimal; no compiler toolchain required.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# google-cloud-storage is the only third-party runtime dep (for gs:// roots).
# Pinned at a floor; let pip resolve a current-compatible version.
RUN pip install "google-cloud-storage>=2.0"

COPY storage_backend.py view_eval_results.py /app/

# Defaults that make sense in Cloud Run; can be overridden at deploy time.
# google-auth resolves ~/.config/... so HOME must be a writable dir owned
# by the runtime user, and XDG_CACHE_HOME pre-empts any stray cache writes
# under a non-writable HOME.
ENV VIEWER_HOST=0.0.0.0 \
    VIEWER_GCS_ONLY=1 \
    PORT=8080 \
    HOME=/home/viewer \
    XDG_CACHE_HOME=/tmp

# Run as a non-root user for defence-in-depth. --create-home gives the
# user a real, writable HOME so google-auth/google-cloud-storage can
# resolve ADC config paths without permission errors at startup.
RUN groupadd --system --gid 10001 viewer \
 && useradd  --system --uid 10001 --gid 10001 \
             --home-dir /home/viewer --create-home --shell /sbin/nologin viewer \
 && chown -R viewer:viewer /app /home/viewer

USER viewer

EXPOSE 8080

# argparse reads env vars: VIEWER_ROOT, PORT, VIEWER_HOST, VIEWER_GCS_ONLY.
CMD ["python", "/app/view_eval_results.py"]
