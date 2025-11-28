# GEMINI.md: AI-Powered Project Context

This file provides essential context for AI assistants to understand and interact with the Aetheris Oracle project.

## Project Overview

Aetheris Oracle is a sophisticated, production-ready probabilistic forecasting engine for cryptocurrency prices. Written in Python, it uses a hybrid approach, combining traditional quantitative models ("Legacy") with state-of-the-art (SOTA) neural models inspired by recent academic research.

The core purpose is not to predict a single price point, but to generate a *distribution* of likely price paths, allowing for robust risk assessment and uncertainty quantification.

**Key Technologies:**
*   **Backend:** Python 3.11+
*   **API:** FastAPI
*   **ML/Scientific Computing:** PyTorch, NumPy, SciPy
*   **Data Sources:** `ccxt`, `yfinance`, Deribit (free public data)
*   **Core Architecture:** A modular pipeline that processes data, applies forecasting models, calibrates results, and serves them via a CLI, Python API, or FastAPI server.

**Architectural Highlights:**
*   **Hybrid Model System:** The engine can run in "Legacy" mode (using baseline models like AR/GRU) or "SOTA" mode (using advanced models like Neural Rough Volatility, Flow Matching, MambaTS). This is controlled via environment variables.
*   **Regime Awareness:** The model adapts its forecasts based on detected market conditions (e.g., calm, volatile, crisis).
*   **Explainability:** The system can attribute risk factors, explaining *why* a forecast is what it is.

## Current Status & Key Issues

The project is mature and well-documented. However, there is a **CRITICAL** bug that is the main focus of current development:

*   **Full SOTA Component Interaction Bug:** While individual SOTA models work perfectly on their own, combining all 6 of them results in nonsensical forecasts (0% coverage). This indicates a negative interaction effect between the components.
*   **Safe Modes:** The "Legacy" mode and a partial SOTA mode ("NCC + Diff Greeks") are considered stable and production-ready.
*   **See Also:** `docs/FULL_SOTA_INVESTIGATION.md` for a deep dive into this bug.

## Building and Running

### 1. Installation

```bash
# 1. Clone the repo
git clone <repo>
cd price-predicter

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set the PYTHONPATH
export PYTHONPATH=src  # Linux/macOS
$env:PYTHONPATH="src"  # Windows PowerShell
```

### 2. Running a Forecast (CLI)

The CLI is the quickest way to get a forecast.

```bash
# Generate a standard 7-day BTC forecast
python -m aetheris_oracle.cli --asset BTC-USD --horizon 7

# Run a forecast using specific SOTA components and plot the result
python -m aetheris_oracle.cli --asset BTC-USD --horizon 7 --use-ncc --use-diff-greeks --plot
```

### 3. Running the API Server

The project includes a FastAPI server for programmatic access.

```bash
# Start the server
uvicorn aetheris_oracle.server:app --host 0.0.0.0 --port 8000
```

You can then `POST` to the `/forecast` endpoint.

### 4. Running Tests

The project has a comprehensive test suite.

```bash
# Run all tests (quick version)
pytest -q

# Run the full test suite with reporting
python run_all_tests.py

# Run tests for a specific module
pytest tests/test_sota_components.py
```

### 5. Training Models

SOTA models require training. Scripts are provided for this.

```bash
# Train all SOTA components using settings from .env
python scripts/train_all_sota.py

# Train a single component
python -m aetheris_oracle.pipeline.train_sota --component ncc --epochs 10
```

## Development Conventions

*   **Configuration via Environment:** Project behavior, especially the use of SOTA vs. Legacy models, is controlled by environment variables defined in a `.env` file (copied from `.env.example`).
*   **Modular & Hybrid Design:** Code is structured into distinct modules for different parts of the forecast pipeline (e.g., `trend`, `volatility`, `jump`). Each logical component has both a `legacy` and a `sota` implementation, which can be toggled.
*   **Extensive Documentation:** The `docs/` directory is well-maintained and serves as the primary source of truth for design decisions, critiques, and implementation status. Refer to it often.
*   **Rigorous Testing:** The project aims for high test coverage. New features should be accompanied by tests.
*   **Utility Scripts:** The `scripts/` directory contains crucial scripts for validation, comparison, training, and debugging.
*   **Known Issues Section:** The `README.md` contains a detailed "Known Issues" section that should be consulted before starting work.
