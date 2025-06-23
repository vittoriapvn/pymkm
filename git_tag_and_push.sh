#!/bin/bash

# === USAGE ===
# bash git_tag_and_push.sh 0.3.0 "Add SMK 2023 full support"

if [[ -z "$1" ]]; then
    echo "❌ ERROR: You must provide a version number (e.g., 0.3.0)"
    exit 1
fi

VERSION="$1"
TAG="v$VERSION"
COMMIT_MESSAGE=${2:-"Release $VERSION"}
CHANGELOG_FILE="CHANGELOG.md"

# === TAG CREATION ===
if git rev-parse "$TAG" >/dev/null 2>&1; then
    echo "⚠️ Tag $TAG already exists. Skipping tag creation."
else
    echo "🏷️ Creating annotated tag $TAG..."
    git tag -a "$TAG" -m "$COMMIT_MESSAGE"
    git push origin "$TAG"
fi

# === COMMIT & PUSH ===
if [[ -n $(git status --porcelain) ]]; then
    echo "🔄 Staging all changes..."
    git add .

    echo "✅ Creating commit: $COMMIT_MESSAGE"
    git commit -m "$COMMIT_MESSAGE"
else
    echo "ℹ️ No changes to commit."
fi

echo "🚀 Pushing to origin/main..."
git push origin main

# === UPDATE CHANGELOG ===
DATE=$(date +%Y-%m-%d)
CHANGELOG_ENTRY="\n## [$VERSION] - $DATE\n- $COMMIT_MESSAGE"

if grep -q "\[$VERSION\]" "$CHANGELOG_FILE"; then
    echo "ℹ️ CHANGELOG already contains entry for $VERSION"
else
    echo "📝 Appending to $CHANGELOG_FILE..."
    echo -e "$CHANGELOG_ENTRY" >> "$CHANGELOG_FILE"
    git add "$CHANGELOG_FILE"
    git commit -m "docs: update CHANGELOG.md for $TAG"
    git push origin main
fi

echo "✅ All done for version $VERSION"