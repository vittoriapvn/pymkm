#!/bin/bash
# Remove previously tracked files that are now in .gitignore

echo "⚠️ This will remove all files from Git tracking that are now ignored by .gitignore."

read -p "Do you want to proceed? (y/n): " confirm
if [[ "$confirm" != "y" ]]; then
    echo "❌ Aborted by user."
    exit 1
fi

echo "🔍 Running: git rm -r --cached ."
git rm -r --cached .

echo "➕ Re-adding files based on updated .gitignore"
git add .

echo "📝 Committing cleanup"
git commit -m "cleanup: remove ignored files from repo"

echo "🚀 Pushing changes to origin/main"
git push origin main

echo "✅ Cleanup complete!"