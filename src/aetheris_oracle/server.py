import logging
import os

import uvicorn
from dotenv import load_dotenv

from .service import create_app

# Load environment variables from .env file
load_dotenv()


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    # Load optional API key from environment
    api_key = os.getenv("AETHERIS_API_KEY")
    if api_key:
        logging.info("API key authentication enabled")
    else:
        logging.info("Running without API key authentication")

    # Load server configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "false").lower() == "true"

    app = create_app(api_key=api_key)
    uvicorn.run(app, host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()
