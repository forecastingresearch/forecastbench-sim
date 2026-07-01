# llm_baseline/

Vendored from https://github.com/bigai-ai/civrealm-llm-baseline (BaseLang / Mastaba
LLM agents from the CivRealm paper), then ported to run on Google Gemini via our
litellm wrapper (`src/civrealm/evaluation/models.py`) with the stale Azure-OpenAI /
langchain / Pinecone dependencies stubbed out. See `docs/forecast_uplift_study.md`.
