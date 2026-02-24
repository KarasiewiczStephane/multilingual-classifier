"""Application entry point for the multilingual classifier API."""

import uvicorn

from src.utils.config import load_config
from src.utils.logger import setup_logger


def main() -> None:
    """Start the FastAPI server with settings from config."""
    config = load_config()
    setup_logger("src", level=config.logging.level)

    uvicorn.run(
        "src.api.app:app",
        host=config.api.host,
        port=config.api.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
