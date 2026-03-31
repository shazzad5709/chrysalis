# Chrysalis

Repository scaffold for the Chrysalis behavioral regression testing tool.

## Development Setup

This repository now uses `uv` as its primary Python project manager.

```zsh
uv sync --dev
uv run python -m spacy download en_core_web_sm
uv run python -c "import nltk; nltk.download('words')"
```

If you prefer to activate the environment directly:

```zsh
source .venv/bin/activate
```
