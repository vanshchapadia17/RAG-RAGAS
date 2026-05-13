# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`RAG-RAGAS` — a Retrieval-Augmented Generation project intended to be evaluated with RAGAS. The codebase is at scaffolding stage: `template.py` generated the directory layout, and most module files (`src/components/*.py`, `src/pipeline/*.py`, `src/utils.py`, `app.py`) are intentionally empty stubs waiting to be filled in. Only `src/logger.py` and `src/exception.py` are implemented.

When asked to "add" functionality, prefer filling in these existing empty files over creating new modules — the scaffold dictates where things go.

## Environment & commands

The project is installed as an editable package (`-e .` in `requirements.txt`), so `src` is importable as a top-level package.

```bash
# create/activate venv (Windows bash)
python -m venv venv
source venv/Scripts/activate

# install deps + editable package
pip install -r requirements.txt

# regenerate the scaffold (idempotent — won't overwrite non-empty files)
python template.py

# run the Flask app (entry point — currently empty)
python app.py
```

No test runner, linter, or formatter is configured yet. There is no Makefile, tox config, or pre-commit setup.

## Architecture

The layout follows the standard "components + pipelines" ML project pattern (à la Krish Naik's template), adapted for RAG:

- **`src/components/`** — single-responsibility building blocks:
  - `data_ingestion.py` — load source documents (PDFs via `pypdf`/`pymupdf`, DOCX via `python-docx`)
  - `data_transformation.py` — chunking + embedding (sentence-transformers) + FAISS index construction
  - `model_trainer.py` — in a RAG context this typically wires the retriever + LLM chain (LangChain + `langchain-groq`) rather than training weights
- **`src/pipeline/`** — orchestration that composes components:
  - `train_pipeline.py` — end-to-end index build
  - `predict_pipeline.py` — query-time retrieval + generation, called by `app.py`
- **`app.py`** — Flask + flask-cors HTTP entry point (per `requirements.txt`); calls into `predict_pipeline`
- **`src/utils.py`** — shared helpers (e.g. `dill` save/load for serialized objects)

### Cross-cutting conventions

**Logging** (`src/logger.py`): import `from src.logger import logging` and use the module-level `logging` object directly — do **not** call `getLogger(__name__)`. The module configures `basicConfig` at import time with `force=True`, writing to `logs/<MM_DD_YYYY_HH_MM_SS>.log`. Note: `os.makedirs(logs_path, ...)` is called on the full file path (not the directory), which is a known quirk of this template — preserve the behavior unless explicitly fixing it.

**Exceptions** (`src/exception.py`): wrap errors in `CustomException` so the script name and line number are captured. Standard usage inside any component:

```python
import sys
from src.exception import CustomException
from src.logger import logging

try:
    ...
except Exception as e:
    raise CustomException(e, sys)
```

The `sys` module **must** be passed as the second argument — `CustomException.__init__` calls `error_detail.exc_info()` on it to extract the traceback frame.

## Notes for future work

- The README is a stub (`"# RAG-RAGAS"`). Don't treat its absence of content as missing context — there genuinely is none yet.
- `.github/workflows/` exists but contains only `.gitkeep`; no CI is wired up.
- RAGAS itself is **not** yet in `requirements.txt` despite being in the project name — add it when evaluation work begins.
