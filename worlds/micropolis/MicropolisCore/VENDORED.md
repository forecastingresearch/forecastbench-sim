# Vendored MicropolisCore

This directory is a trimmed copy of Fabio Rocha's fork of MicropolisCore, the engine the
Micropolis world runs (`micropolis_world.module_globals.ENGINE_ROOT`).

- Repository: https://github.com/fdrocha/MicropolisCore (fork of SimHacker/MicropolisCore)
- Branch: `fbs`
- Commit: `4ba571dd9a571e674c919b17a5577ff61fbffe2c` (2026-11-04)
- Vendored: November 10, 2026

`engine_commit_note()` reads the commit line above, so update it when re-vendoring.

## What was left out

Files tracked upstream but not needed to build or run the engine CLIs:

- `documentation/`, `images/`: docs, talks, slides, diagrams
- `content/` except `content/micropolis/cities/` (the engine build preloads the cities)
- `apps/micropolis/static/`, `apps/micropolis/website/`: website assets and pages
- `skills/`, `apps/soul-angel/`: agent and design docs
- `.github/`: upstream CI, which does not run from a subdirectory
- `apps/vitamoospace/static/data`: a symlink into the omitted VitaMoo content
- `.vscode/`

Everything else is byte-identical to the upstream commit, including the prebuilt engine in
`apps/micropolis/src/lib/micropolisengine.{js,wasm,data}`, so running the sims needs only Node >=20
and pnpm (no Emscripten):

```sh
cd worlds/micropolis/MicropolisCore
pnpm install --frozen-lockfile
```

Rebuilding the engine (`pnpm run build:engine`) needs Emscripten; see `README.md`.

## Re-vendoring

From a clean checkout of the fork at the new commit (`$SRC`), with this directory as `$DEST`:

```sh
git -C "$SRC" ls-files | grep -v -E '^(documentation|images|content|skills|\.vscode|\.github)/|^apps/(soul-angel|micropolis/static|micropolis/website)/|^apps/vitamoospace/static/data$' > keep.txt
git -C "$SRC" ls-files 'content/micropolis/cities/*' >> keep.txt
git -C "$SRC" archive HEAD $(cat keep.txt) | tar -x -C "$DEST"
```

Delete files that disappeared upstream, keep this file, update the commit line, and rerun
`pnpm install --frozen-lockfile`.
