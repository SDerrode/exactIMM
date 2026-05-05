# Wiki sources

These markdown files are the source for the project's GitHub Wiki.

## Pages

- `Home.md` — landing page
- `Installation.md` — install / verify
- `Tutorial.md` — end-to-end example
- `Paper-Reproduce.md` — reproduce the paper experiments
- `API-Overview.md` — module/class reference
- `GUI-Guide.md` — PyQt6 interface guide
- `Citing.md` — BibTeX entries

## Pushing to the GitHub Wiki

GitHub stores wikis in a *separate* git repository at
`<repo>.wiki.git`. To sync the markdown sources here to the wiki:

```bash
# One-time: clone the wiki next to the main repo
cd ..
git clone https://github.com/SDerrode/exactIMM.wiki.git

# Each time you update wiki/*.md, rsync and push.
# IMPORTANT: --exclude='.git/' is required — without it, rsync --delete
# would wipe the wiki repo's .git/ directory (destination matches source,
# and the local wiki/ source has no .git/). --exclude='.gitignore' keeps
# the wiki free of a file that's only meaningful in the main repo.
WIKI_DIR=exactIMM.wiki
# Guard: re-clone if the directory exists but its .git/ is missing
# (e.g. wiped by a previous bad rsync).
if [ -d "$WIKI_DIR" ] && [ ! -d "$WIKI_DIR/.git" ]; then
    rm -rf "$WIKI_DIR"
    git clone https://github.com/SDerrode/exactIMM.wiki.git "$WIKI_DIR"
fi
rsync -av --delete \
    --exclude='.git/' \
    --exclude='.gitignore' \
    --exclude='README.md' \
    exactIMM/wiki/ "$WIKI_DIR/"
cd "$WIKI_DIR"
git add . && git commit -m "Update wiki" && git push
```

If you prefer not to maintain a wiki, you can simply delete the
`wiki/` directory; everything important is also in the main `README.md`.
