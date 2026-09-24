# Offline benchmark package

Follow the root CLAUDE.md. Use `uv run --offline --no-project --python <venv>/bin/python` for Python commands. Read README.md and docs/SEMANTICS.md. Tests use invented fixtures only; never launch engines, model calls or production scripts implicitly. Preserve native mathematical/parser bodies and source attribution. Paper selection, normalization and frozen data belong outside this package. Do not bundle fbsim_core: packages/fbsim-core owns it. Publication requires separate approval.
