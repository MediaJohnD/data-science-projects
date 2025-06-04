#!/bin/bash
set -e

docker build -t dspipeline-api -f docker/Dockerfile .
# Push to registry (replace with your registry URL)
docker push dspipeline-api || echo "Skipping push in local environment"
