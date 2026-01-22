"""Application entry point."""

import uvicorn

from src.config.settings import get_settings
from src.logging.setup import setup_logging


def main() -> None:
    """Run the application."""
    settings = get_settings()

    # Setup logging
    json_format = not settings.is_development
    setup_logging(log_level=settings.log_level, json_format=json_format)

    # Run server
    uvicorn.run(
        "src.api.app:app",
        host=settings.app.server.host,
        port=settings.app.server.port,
        workers=settings.app.server.workers,
        reload=settings.is_development,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
