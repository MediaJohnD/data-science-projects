#!/usr/bin/env bash
set -e
# Configure Git remote if not already present
REMOTE_NAME="origin"
REMOTE_URL="https://github.com/MediaJohnD/data-science-projects.git"

if git rev-parse --git-dir > /dev/null 2>&1; then
  if git remote | grep -q "$REMOTE_NAME"; then
    echo "Remote '$REMOTE_NAME' already configured:" 
    git remote -v
  else
    git remote add "$REMOTE_NAME" "$REMOTE_URL"
    git remote -v
  fi
else
  echo "This directory is not a Git repository. Initialize with git init first." >&2
  exit 1
fi
