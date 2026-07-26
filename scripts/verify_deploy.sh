#!/usr/bin/env bash
# Post-deploy smoke test for the Milestone 7 Render + Vercel deployment.
#
# Usage: scripts/verify_deploy.sh <backend-url> <frontend-url>
#   e.g. scripts/verify_deploy.sh https://sentinelml-backend.onrender.com https://sentinelml.vercel.app
#
# Checks the same things CI's `docker` job checks against the local Compose
# stack (see .github/workflows/ci.yml), against the real deployed services:
# backend liveness/readiness, OpenAPI schema, and that the frontend serves
# the SPA and can reach the backend through its configured API base URL.

set -euo pipefail

if [ $# -ne 2 ]; then
  echo "usage: $0 <backend-url> <frontend-url>" >&2
  exit 1
fi

backend_url="${1%/}"
frontend_url="${2%/}"
failures=0

check() {
  local description="$1" url="$2" expected_status="$3"
  local status
  status=$(curl -s -o /tmp/verify_deploy_body -w '%{http_code}' "$url" || echo "000")
  if [ "$status" = "$expected_status" ]; then
    echo "OK   $description ($url -> $status)"
  else
    echo "FAIL $description ($url -> $status, expected $expected_status)"
    cat /tmp/verify_deploy_body
    echo
    failures=$((failures + 1))
  fi
}

echo "== Backend: $backend_url =="
check "liveness (/api/v1/health)" "$backend_url/api/v1/health" 200
check "readiness (/api/v1/ready)" "$backend_url/api/v1/ready" 200
check "OpenAPI docs (/docs)" "$backend_url/docs" 200
check "OpenAPI schema (/openapi.json)" "$backend_url/openapi.json" 200
check "model metadata (/api/v1/model)" "$backend_url/api/v1/model" 200

echo
echo "== Frontend: $frontend_url =="
check "SPA root (/)" "$frontend_url/" 200
check "client-side route falls back to SPA (/predict)" "$frontend_url/predict" 200

echo
if [ "$failures" -eq 0 ]; then
  echo "All checks passed."
else
  echo "$failures check(s) failed." >&2
  exit 1
fi
