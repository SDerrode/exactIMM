#!/usr/bin/env bash
# scripts/sync_wiki.sh
# ====================
# Push the local wiki/ markdown sources to the GitHub Wiki.
#
# Prerequisite: the wiki repo must exist on GitHub. To bootstrap it:
#   1. Open https://github.com/SDerrode/exactIMM/wiki
#   2. Click "Create the first page"
#   3. Save any placeholder page (it will be overwritten by this script)
#
# Then run this script from anywhere; it will:
#   - clone (or pull) the wiki repo into ../exactIMM.wiki/
#   - rsync wiki/*.md (except wiki/README.md) into it
#   - commit + push
#
# Usage:
#   ./scripts/sync_wiki.sh
#   ./scripts/sync_wiki.sh --dry-run   # show what would be pushed

set -euo pipefail

REPO_OWNER="SDerrode"
REPO_NAME="exactIMM"
WIKI_URL="https://github.com/${REPO_OWNER}/${REPO_NAME}.wiki.git"

# Locate this repo's root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
WIKI_SRC="${REPO_ROOT}/wiki"
WIKI_DIR="${REPO_ROOT}/../${REPO_NAME}.wiki"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

# 1. Sanity checks
if [[ ! -d "${WIKI_SRC}" ]]; then
    echo "ERROR: ${WIKI_SRC} not found" >&2
    exit 1
fi

# 2. Clone or update the wiki repo
# If WIKI_DIR exists but lacks .git/ (e.g. previous run wiped it), wipe and re-clone.
if [[ -d "${WIKI_DIR}" && ! -d "${WIKI_DIR}/.git" ]]; then
    echo "WARN: ${WIKI_DIR} exists but is not a git repo — re-cloning."
    rm -rf "${WIKI_DIR}"
fi

if [[ ! -d "${WIKI_DIR}" ]]; then
    echo "Cloning ${WIKI_URL} → ${WIKI_DIR}"
    if ! git clone "${WIKI_URL}" "${WIKI_DIR}"; then
        cat >&2 <<EOF

ERROR: failed to clone the wiki repository.

The most likely reason is that the wiki has never been initialised on GitHub.
Please:
  1. Open https://github.com/${REPO_OWNER}/${REPO_NAME}/wiki
  2. Click "Create the first page" and save any placeholder content
  3. Re-run this script

EOF
        exit 2
    fi
else
    echo "Updating existing ${WIKI_DIR}"
    git -C "${WIKI_DIR}" pull --quiet
fi

# 3. Sync content. CRITICAL: exclude .git/ (otherwise --delete wipes the
#    wiki's git repo since it doesn't exist in the source) and the
#    wiki/README.md meta file (only useful in the main repo).
echo "Syncing ${WIKI_SRC}/ → ${WIKI_DIR}/"
rsync -av --delete \
    --exclude='.git/' \
    --exclude='.gitignore' \
    --exclude='README.md' \
    "${WIKI_SRC}/" "${WIKI_DIR}/"

# 4. Commit + push
cd "${WIKI_DIR}"
if git diff --quiet && git diff --cached --quiet; then
    echo "No changes to push."
    exit 0
fi

git status --short
if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "(dry-run: stopping here)"
    exit 0
fi

git add -A
git commit -m "Sync wiki from main repo ($(date +%Y-%m-%d))"
git push

echo
echo "Done. Wiki updated at https://github.com/${REPO_OWNER}/${REPO_NAME}/wiki"
