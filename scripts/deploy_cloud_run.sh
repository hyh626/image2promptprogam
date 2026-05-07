#!/usr/bin/env bash
# Build the viewer image and deploy it to Cloud Run.
#
# The Cloud Run service runs as the service account passed via
# --service-account; that SA needs roles/storage.objectViewer on the bucket
# referenced by --bucket. The container is configured to refuse local roots
# (VIEWER_GCS_ONLY=1), so --bucket must be a gs:// URI.
#
# Required:
#   --bucket gs://bucket/prefix          GCS root the viewer will browse
#   --service-account EMAIL              runtime SA for the Cloud Run service
#
# Optional:
#   --project PROJECT_ID                 default: $GOOGLE_CLOUD_PROJECT or
#                                        gcloud's active project
#   --region REGION                      default: us-central1
#   --service NAME                       default: image2prompt-viewer
#   --image URL                          default: gcr.io/PROJECT/SERVICE
#   --memory SIZE                        default: 512Mi
#   --cpu N                              default: 1
#   --concurrency N                      default: 40
#   --max-instances N                    default: 4
#   --min-instances N                    default: 0
#   --gcs-cache-ttl SECONDS              default: 30
#   --allow-unauthenticated              make the URL public (default: off)
#   --iap                                turn on Identity-Aware Proxy on
#                                        the Cloud Run service so a
#                                        browser session can sign in with
#                                        Google IAM and reach the URL
#                                        without a bearer token. Mutually
#                                        exclusive with
#                                        --allow-unauthenticated.
#   --iap-member EMAIL                   grant this principal
#                                        roles/iap.httpsResourceAccessor
#                                        on the Cloud Run service.
#                                        Repeatable. Accepts the gcloud
#                                        IAM member syntax, e.g.
#                                        user:alice@example.com,
#                                        group:eng@example.com,
#                                        domain:example.com.
#   --skip-build                         reuse the existing image; skip
#                                        gcloud builds submit
#   --dry-run                            print the gcloud commands without
#                                        running them
#
# Examples:
#   ./scripts/deploy_cloud_run.sh \
#     --project my-proj --region us-central1 \
#     --service image2prompt-viewer \
#     --bucket gs://image2promptdata/experiments \
#     --service-account viewer-runtime@my-proj.iam.gserviceaccount.com
#
#   ./scripts/deploy_cloud_run.sh ... --allow-unauthenticated   # public URL
#
#   # IAP-protected URL openable in a browser by named users:
#   ./scripts/deploy_cloud_run.sh ... \
#     --iap \
#     --iap-member user:alice@example.com \
#     --iap-member group:eng@example.com

set -euo pipefail

usage() {
  sed -n '2,59p' "$0" | sed 's/^# \{0,1\}//'
}

PROJECT="${GOOGLE_CLOUD_PROJECT:-}"
REGION="us-central1"
SERVICE="image2prompt-viewer"
IMAGE=""
BUCKET=""
SERVICE_ACCOUNT=""
MEMORY="512Mi"
CPU="1"
CONCURRENCY="40"
MAX_INSTANCES="4"
MIN_INSTANCES="0"
GCS_CACHE_TTL="30"
ALLOW_UNAUTHENTICATED=0
ENABLE_IAP=0
IAP_MEMBERS=()
SKIP_BUILD=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project) PROJECT="$2"; shift 2 ;;
    --region) REGION="$2"; shift 2 ;;
    --service) SERVICE="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --bucket) BUCKET="$2"; shift 2 ;;
    --service-account) SERVICE_ACCOUNT="$2"; shift 2 ;;
    --memory) MEMORY="$2"; shift 2 ;;
    --cpu) CPU="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --max-instances) MAX_INSTANCES="$2"; shift 2 ;;
    --min-instances) MIN_INSTANCES="$2"; shift 2 ;;
    --gcs-cache-ttl) GCS_CACHE_TTL="$2"; shift 2 ;;
    --allow-unauthenticated) ALLOW_UNAUTHENTICATED=1; shift ;;
    --iap) ENABLE_IAP=1; shift ;;
    --iap-member) IAP_MEMBERS+=("$2"); shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$BUCKET" ]]; then
  echo "error: --bucket gs://... is required" >&2
  exit 2
fi
if [[ "$BUCKET" != gs://* ]]; then
  echo "error: --bucket must be a gs:// URI; got $BUCKET" >&2
  exit 2
fi
if [[ -z "$SERVICE_ACCOUNT" ]]; then
  echo "error: --service-account EMAIL is required" >&2
  exit 2
fi
if (( ENABLE_IAP )) && (( ALLOW_UNAUTHENTICATED )); then
  echo "error: --iap and --allow-unauthenticated are mutually exclusive" >&2
  exit 2
fi
if (( ${#IAP_MEMBERS[@]} > 0 )) && ! (( ENABLE_IAP )); then
  echo "error: --iap-member requires --iap" >&2
  exit 2
fi
if [[ -z "$PROJECT" ]]; then
  PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
  if [[ -z "$PROJECT" || "$PROJECT" == "(unset)" ]]; then
    echo "error: --project is required (or set GOOGLE_CLOUD_PROJECT, " \
         "or gcloud config set project ...)" >&2
    exit 2
  fi
fi
if [[ -z "$IMAGE" ]]; then
  IMAGE="gcr.io/${PROJECT}/${SERVICE}"
fi

run() {
  if (( DRY_RUN )); then
    printf '+ %q' "$1"
    shift
    for a in "$@"; do printf ' %q' "$a"; done
    printf '\n'
  else
    "$@"
  fi
}

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if (( SKIP_BUILD )); then
  echo "==> skipping build; using existing image $IMAGE"
else
  echo "==> building $IMAGE via Cloud Build (project=$PROJECT)"
  run gcloud builds submit "$REPO_ROOT" \
    --tag "$IMAGE" \
    --project "$PROJECT" \
    --quiet
fi

DEPLOY_ARGS=(
  run deploy "$SERVICE"
  --image "$IMAGE"
  --project "$PROJECT"
  --region "$REGION"
  --service-account "$SERVICE_ACCOUNT"
  --memory "$MEMORY"
  --cpu "$CPU"
  --concurrency "$CONCURRENCY"
  --max-instances "$MAX_INSTANCES"
  --min-instances "$MIN_INSTANCES"
  --port 8080
  --set-env-vars "VIEWER_ROOT=${BUCKET},VIEWER_GCS_ONLY=1,VIEWER_GCS_CACHE_TTL=${GCS_CACHE_TTL}"
)
if (( ALLOW_UNAUTHENTICATED )); then
  DEPLOY_ARGS+=(--allow-unauthenticated)
else
  DEPLOY_ARGS+=(--no-allow-unauthenticated)
fi
if (( ENABLE_IAP )); then
  # Cloud Run Gen2 supports a built-in IAP integration: --iap on the
  # service makes IAP the only path to the URL, and signed-in browser
  # users with roles/iap.httpsResourceAccessor can reach the page
  # directly. (Older gcloud versions don't know this flag; the failure
  # message is informative.)
  DEPLOY_ARGS+=(--iap)
fi

echo "==> deploying $SERVICE to Cloud Run in $REGION as $SERVICE_ACCOUNT"
run gcloud "${DEPLOY_ARGS[@]}"

if (( ENABLE_IAP )) && (( ${#IAP_MEMBERS[@]} > 0 )); then
  echo "==> granting roles/iap.httpsResourceAccessor on $SERVICE to ${#IAP_MEMBERS[@]} member(s)"
  for member in "${IAP_MEMBERS[@]}"; do
    run gcloud run services add-iam-policy-binding "$SERVICE" \
      --project "$PROJECT" --region "$REGION" \
      --member "$member" \
      --role roles/iap.httpsResourceAccessor \
      --quiet
  done
fi

if ! (( DRY_RUN )); then
  URL="$(gcloud run services describe "$SERVICE" \
    --project "$PROJECT" --region "$REGION" \
    --format='value(status.url)' 2>/dev/null || true)"
  if [[ -n "$URL" ]]; then
    echo "==> service URL: $URL"
    if (( ENABLE_IAP )); then
      echo "    (IAP-protected; open in a browser as a signed-in member with"
      echo "     roles/iap.httpsResourceAccessor on this service)"
      if ! (( ${#IAP_MEMBERS[@]} > 0 )); then
        echo "    (no --iap-member given; grant access with:"
        echo "       gcloud run services add-iam-policy-binding $SERVICE \\"
        echo "         --project $PROJECT --region $REGION \\"
        echo "         --member user:you@example.com \\"
        echo "         --role roles/iap.httpsResourceAccessor )"
      fi
    elif ! (( ALLOW_UNAUTHENTICATED )); then
      echo "    (IAM-protected; reach it with:"
      echo "       curl -H \"Authorization: Bearer \$(gcloud auth print-identity-token)\" $URL )"
    fi
  fi
fi
