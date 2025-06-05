#!/bin/bash
set -e

tag=${1:-latest}
docker build -t dspipeline-api:${tag} -f docker/Dockerfile .
# Push to registry (replace with your registry URL)
docker push dspipeline-api:${tag} || echo "Skipping push in local environment"
