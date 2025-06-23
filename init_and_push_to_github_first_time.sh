#!/bin/bash
# Initialize Git repository and push to GitHub (run only once)

GITHUB_USERNAME="BeppeMagro"

echo "[1/4] Initializing Git repository..."
git init

echo "[2/4] Adding all files..."
git add .

echo "[3/4] Committing..."
git commit -m "Initial commit of pyMKM"

echo "[4/4] Adding remote and pushing to GitHub..."
git remote add origin https://github.com/${GITHUB_USERNAME}/pymkm.git
git branch -M main
git push -u origin main

echo "✅ Done! This script should only be run ONCE when setting up the project."