# Repository Guidelines

## Project Structure & Module Organization
- `webapp.py`: Flask app entry (routes, JSON/HTML responses).
- `app.py`: CLI analysis demo (non-server entry).
- `data_fetcher.py`, `stock_analyzer.py`, `ai_analysis.py`, `config.py`: core domain logic and settings.
- `templates/`, `static/`, `assets/`: UI templates, static JS/CSS, and images.
- `tests/`: system tests using Selenium + unittest; `requirements-test.txt` lists extras.
- `docs/ARCHITECTURE.md`: detailed architecture and endpoints overview.

## Build, Test, and Development Commands
- Create venv: `python3 -m venv .venv && source .venv/bin/activate`
- Install deps: `pip install -r requirements.txt`
- Run dev server (port 5000): `./run.sh` or `FLASK_APP=webapp.py python -m flask run`
- Run on port 8080 (tests expect): `FLASK_APP=webapp.py python -m flask run --port 8080`
- Install test deps: `pip install -r tests/requirements-test.txt`
- Run tests: `python -m unittest tests/test_system.py` (or `pytest -q`)

## Coding Style & Naming Conventions
- Python: PEP 8, 4-space indentation, snake_case for functions/vars, CapWords for classes.
- Files: modules lowercase with underscores (e.g., `data_fetcher.py`).
- Templates: `templates/*.html`; Static: `static/js`, `static/css`.
- Docstrings for public functions; keep functions small and pure where possible.

## Testing Guidelines
- Frameworks: unittest system test with Selenium; pytest compatible.
- Naming: tests in `tests/` as `test_*.py`; helper utilities go in `tests/utils.py`.
- Setup: ensure Chrome is available; `webdriver-manager` installs drivers automatically.
- Coverage: add unit tests for new logic in `data_fetcher.py`, `stock_analyzer.py`, `ai_analysis.py`.

## Commit & Pull Request Guidelines
- Commits: present tense, concise subject, body with rationale (e.g., "Add RSI threshold to analyzer").
- PRs: clear description, linked issue, before/after screenshots for UI, steps to test, and note any config/env changes.
- Requirements: run tests locally; update `docs/ARCHITECTURE.md` when modifying routes or APIs.

## Security & Configuration Tips
- Do not commit secrets. Override `SECRET_KEY` via env var in production.
- Network calls hit Yahoo Finance; consider adding caching/mocking for new unit tests.

## Docker Workflow
- Build image (prod): `docker build -t portfolio:latest .`
- Run container (port 8080): `docker run --rm -p 8080:8080 -e FLASK_APP=webapp.py portfolio:latest`
- Dev with live reload: `docker compose up --build` (bind-mounts source; reloads on changes).
- Enter shell: `docker run -it --entrypoint bash portfolio:latest`
- Run tests inside container:
  - Start app in one terminal: `docker run --rm -p 8080:8080 portfolio:latest`
  - In another: `docker run --rm --network host -v "$PWD":/app -w /app portfolio:latest bash -lc "pip install -r tests/requirements-test.txt && python -m unittest tests/test_system.py"`
