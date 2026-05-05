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
`<repo>.wiki.git`. The helper script `scripts/sync_wiki.sh` clones (or
pulls) `../exactIMM.wiki/`, rsyncs the markdown sources into it
(safely — preserves the wiki's `.git/`), commits and pushes:

```bash
./scripts/sync_wiki.sh             # sync + push
./scripts/sync_wiki.sh --dry-run   # preview what would change
```

Bootstrapping (first time only): the wiki repo must exist on GitHub.
If you've never opened it, visit
<https://github.com/SDerrode/exactIMM/wiki>, click *Create the first
page*, save any placeholder, then run the script.

If you prefer not to maintain a wiki, you can simply delete the
`wiki/` directory; everything important is also in the main `README.md`.
