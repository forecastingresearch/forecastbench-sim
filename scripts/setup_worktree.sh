#!/usr/bin/env bash
# setup_worktree.sh [--quiet] — give this checkout its data/ and credentials/ links, and the FreeCiv v3 shortcut
# worlds/freeciv/fbsim_v3/fleet/runs -> data/freeciv/fbsim_v3/runs; idempotent.
# A linked worktree points both at what the main checkout has (a symlink's target or a real directory).
# The main checkout points them at $FBSIM_LOCAL/{data,credentials} (default: fbsim-local next to the main checkout).
# Run by .githooks/post-checkout once `git config core.hooksPath .githooks` is set.
set -euo pipefail
QUIET=0; [ "${1:-}" = "--quiet" ] && QUIET=1
say() { [ $QUIET = 1 ] || echo "$@"; }
ROOT="$(git rev-parse --show-toplevel)"
MAIN="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")"
LOCAL="${FBSIM_LOCAL:-$(dirname "$MAIN")/fbsim-local}"
for name in data credentials; do
  link="$ROOT/$name"
  if [ -L "$link" ] && [ ! -e "$link" ]; then echo "setup_worktree: $name -> $(readlink "$link") is dangling" >&2; continue; fi
  if [ -e "$link" ]; then say "$name: present ($( [ -L "$link" ] && readlink "$link" || echo directory))"; continue; fi
  if [ "$ROOT" != "$MAIN" ]; then
    [ -e "$MAIN/$name" ] || { echo "setup_worktree: the main checkout $MAIN has no $name/; run this script there first" >&2; continue; }
    target="$(cd "$MAIN/$name" && pwd -P)"
  else
    target="$LOCAL/$name"
    [ -d "$target" ] || { echo "setup_worktree: $target does not exist; create it (or set FBSIM_LOCAL) and rerun" >&2; continue; }
  fi
  ln -s "$target" "$link"; echo "setup_worktree: $name -> $target"
done
# relative, so it resolves through this checkout's data/ link
runs="$ROOT/worlds/freeciv/fbsim_v3/fleet/runs"
if [ -d "$ROOT/worlds/freeciv/fbsim_v3/fleet" ] && [ ! -L "$runs" ] && [ ! -e "$runs" ]; then
  [ -d "$ROOT/data" ] && mkdir -p "$ROOT/data/freeciv/fbsim_v3/runs"
  ln -s ../../../../data/freeciv/fbsim_v3/runs "$runs"; echo "setup_worktree: worlds/freeciv/fbsim_v3/fleet/runs -> data/freeciv/fbsim_v3/runs"
fi
