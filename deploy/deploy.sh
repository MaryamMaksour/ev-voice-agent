#!/usr/bin/env bash
# Deploy one commit on the GPU server, check every service, and roll back if the check fails.
#
#   DEPLOY_DIR=/opt/ev-voice-agent bash deploy/deploy.sh <git-sha>
#
# GitHub Actions runs this on the server (self-hosted runner, see README "CI/CD"); you can run it by hand too.
# DEPLOY_DIR is the server's clone of the repo, where .env, data/ and logs/ live. Default: this repo.
set -euo pipefail

TARGET="${1:?usage: deploy.sh <git-sha or branch>}"
DIR="${DEPLOY_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$DIR"

if [ ! -f .env ]; then
  echo "no .env in $DIR: copy .env.example to .env and fill it in first" >&2
  exit 1
fi

# the docker compose command for this .env (SERVER_LLM, SERVER_GPU): see deploy/stack.sh
source "$DIR/deploy/stack.sh"
stack_compose
HEALTH_TRIES="${HEALTH_TRIES:-18}"     # x 10 s = 3 minutes for the agent to come up

PREV=$(git rev-parse HEAD)
git fetch --quiet --prune origin
git checkout --quiet --detach "$TARGET"
NEW=$(git rev-parse HEAD)
mkdir -p data/recordings data/calls logs
chmod 777 data/recordings   # the Egress container writes here as its own user

up() {
  local sha
  sha=$(git rev-parse --short HEAD)
  echo "==> starting $sha"
  # running calls finish first: the old agent gets up to 10 minutes (stop_grace_period)
  GIT_SHA="$sha" "${COMPOSE[@]}" up -d --build --remove-orphans
}

healthy() {
  for i in $(seq 1 "$HEALTH_TRIES"); do
    if "${COMPOSE[@]}" exec -T agent python -m agent.check; then
      return 0
    fi
    echo "==> not healthy yet ($i/$HEALTH_TRIES), retrying in 10 s"
    sleep 10
  done
  return 1
}

if [ "$PREV" = "$NEW" ]; then
  echo "==> $NEW is already checked out; rebuilding anyway"
fi

up
if healthy; then
  echo "==> deployed $(git rev-parse --short HEAD)"
  # free disk: untagged leftovers of older builds (tagged versions stay, for a manual rollback)
  docker image prune -f >/dev/null || true
  exit 0
fi

echo "==> health check failed: rolling back to $(git rev-parse --short "$PREV")" >&2
"${COMPOSE[@]}" logs --tail 100 agent >&2 || true
git checkout --quiet --detach "$PREV"
up
healthy || echo "==> the previous version is not healthy either: check the server by hand" >&2
exit 1
