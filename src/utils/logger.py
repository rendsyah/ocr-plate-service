import json
import logging
import sys
import warnings

from loguru import logger

from src.config import get_settings


def serialize(record):
    """
    Custom JSON serializer for structured production logs.
    """
    settings = get_settings()

    subset = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name.lower(),
        "message": record["message"],
        "service": record["extra"].get("service", settings.app_name),
        "environment": record["extra"].get("environment", settings.app_env),
        "context": record["extra"].get("context", record["name"]),
        "traceId": record["extra"].get("traceId"),
    }

    # Include HTTP request/response data if available
    if "req" in record["extra"]:
        subset["req"] = record["extra"]["req"]
    if "res" in record["extra"]:
        subset["res"] = record["extra"]["res"]
    if "responseTime" in record["extra"]:
        subset["responseTime"] = record["extra"]["responseTime"]

    # Include any other extra fields
    for key, value in record["extra"].items():
        if key not in [
            "service",
            "environment",
            "context",
            "traceId",
            "req",
            "res",
            "responseTime",
            "logger_name",
        ]:
            subset[key] = value

    return json.dumps(subset, default=str)


def setup_app_logging(log_level: str = "INFO"):
    """
    Setup loguru to handle all application logs, with split logic for App and HTTP.
    Also suppresses common library noise.
    """
    # 0. Suppress common library noise
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*ccache.*")

    # Remove all default handlers
    logger.remove()

    # 1. Primary Console Handler (Human Readable)
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    # 2. File Handlers (JSON Format for Production)
    if get_settings().app_env != "development":
        # Application Logic & Errors (Exclude all HTTP access logs)
        logger.add(
            "logs/app.log",
            rotation="10 MB",
            retention="7 days",
            level=log_level,
            filter=lambda record: (
                record["extra"].get("context") != "HTTP"
                and record["extra"].get("logger_name") != "uvicorn.access"
            ),
            format="{extra[serialized]}",
        )

        # HTTP Access Logs (Only our custom middleware logs)
        logger.add(
            "logs/http.log",
            rotation="20 MB",
            retention="3 days",
            level=log_level,
            filter=lambda record: record["extra"].get("context") == "HTTP",
            format="{extra[serialized]}",
        )

    # Bind default context
    logger.configure(
        patcher=lambda record: record["extra"].update(serialized=serialize(record))
    )

    # Intercept standard logging (Uvicorn, etc.)
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            # Find caller from where originated the logged message
            frame, depth = logging.currentframe(), 2
            while frame and frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).bind(
                logger_name=record.name
            ).log(level, record.getMessage())

    # Apply interception
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    for name in ["uvicorn", "uvicorn.access", "uvicorn.error", "fastapi"]:
        _logger = logging.getLogger(name)
        _logger.handlers = [InterceptHandler()]
        _logger.propagate = False
