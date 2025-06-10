#!/bin/bash
# Install the Python package in editable mode
sudo pip install -e $REPO_DIR
sudo pip install opencv-python-headless
exec "$@"