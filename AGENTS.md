# Repository Guidelines

## Project Structure & Module Organization
- Core code in `src/aetheris_oracle`: `pipeline/` orchestration (forecast, calibration, train_sota), `modules/` models (trend, volatility, jump, residual, SOTA variants), `features/` preprocessing, `monitoring/` metrics, `utils/` helpers.
- API/CLI entrypoints: `aetheris_oracle.cli`, `server.py` (FastAPI), and `start.py` as a convenience wrapper.
- Tests live in `tests/` (pipeline, connectors, SOTA, performance); `scripts/` hosts validation, training, and diagnostics; `docs/` contains design notes, project status, and testing reports.

## Build, Test, and Development Commands
```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
$env:PYTHONPATH="src"  # required for all commands
python -m aetheris_oracle.cli --asset BTC-USD --horizon 7 --paths 500
python -m aetheris_oracle.server  # FastAPI service
python start.py --mode service    # alternative entrypoint
python run_all_tests.py           # full suite wrapper
pytest -q                         # quick smoke
python scripts/run_validation.py --recent
python scripts/verify_sota.py     # SOTA config check
```

## Coding Style & Naming Conventions
- Python 3.11+, 4-space indentation, PEP 8; keep functions pure where possible and favor the config/dataclass patterns already used (e.g., `ForecastConfig`).
- Type hints required; prefer explicit return types and `Dict[str, float]` style hints for CLI parsing.
- Naming: files/modules `snake_case.py`, functions/methods `snake_case`, classes `CamelCase`, constants `UPPER_SNAKE`.
- Keep CLI/API flags aligned between `cli.py` and `api_schemas.py`; avoid magic numbers; introduce named constants or config defaults.
- Use dotenv for settings (`.env.example`); never commit `.env`, cached outputs, or `artifacts/` model files.

## Testing Guidelines
- Add or update tests in `tests/` with names `test_*.py::test_*`; keep deterministic seeds when feasible.
- Primary runner: `pytest` (cache disabled via `pytest.ini`); prefer `run_all_tests.py` before PRs to cover integration and performance suites.
- For new SOTA features, include validation paths or component diagnostics (e.g., `scripts/diagnose_components.py`) and note expected coverage/latency impacts.

## Commit & Pull Request Guidelines
- Use imperative, scoped messages (prefer Conventional Commits): `feat: add ncc calibration guard`, `fix: handle missing connector auth`.
- Before PR: ensure `PYTHONPATH=src`, tests/validation scripts above are green, and docs updated when behavior changes (`docs/implementation/PROJECT_STATUS.md`, `README.md` flags).
- PR description should state summary, test commands run, any new config flags, and screenshots/plots when altering outputs or visualization.
- Link issues or TODOs referenced in README/docs; avoid committing large artifacts; reference expected locations instead.
