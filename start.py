"""
Convenience entrypoint to run the FastAPI service, CLI forecast, or offline evaluation.

Usage:
- Service: python start.py --mode service
- CLI:     python start.py --mode cli --asset BTC-USD --horizon 7 --paths 500
- Eval:    python start.py --mode offline_eval

Note: Service configuration (API key, host, port) can be set in .env file
"""

import argparse
from pprint import pprint

from dotenv import load_dotenv

from aetheris_oracle.cli import main as cli_main
from aetheris_oracle.config import ForecastConfig
from aetheris_oracle.pipeline.offline_evaluation import run_walk_forward
from aetheris_oracle.server import main as server_main

# Load environment variables from .env file
load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start Aetheris Oracle service, CLI forecast, or evaluation.")
    parser.add_argument("--mode", choices=["service", "cli", "offline_eval"], default="service", help="Run mode")
    parser.add_argument("--api-key", help="API key for service mode")
    parser.add_argument("--host", default="0.0.0.0", help="Host for service mode")
    parser.add_argument("--port", type=int, default=8000, help="Port for service mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "cli":
        cli_main()
    elif args.mode == "offline_eval":
        result = run_walk_forward([ForecastConfig(horizon_days=2, num_paths=50)])
        print("Offline evaluation summary:")
        pprint(result.coverage_summary())
    else:
        # server_main currently ignores host/port; could be extended to accept them
        server_main()


if __name__ == "__main__":
    main()
