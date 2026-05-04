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
git clone https://github.com/<USER>/fofgss.wiki.git

# Each time you update wiki/*.md, rsync and push
rsync -av --delete --exclude README.md fofgss/wiki/ fofgss.wiki/
cd fofgss.wiki
git add . && git commit -m "Update wiki" && git push
```

If you prefer not to maintain a wiki, you can simply delete the
`wiki/` directory; everything important is also in the main `README.md`.
