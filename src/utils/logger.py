import json
import logging
import sys
import warnings

from loguru import logger

from src.config import get_settings

# Keys that should never be logged or should be masked for security
SENSITIVE_KEYS = {
    "password",
    "token",
    "secret",
    "key",
    "authorization",
    "auth",
    "credential",
    "cookie",
    "signature",
    "api_key",
}


def clean_record_extra(extra_dict):
    """
    Recursively clean or filter sensitive keys from log record extra.
    """
    if not isinstance(extra_dict, dict):
        return extra_dict
    cleaned = {}
    for k, v in extra_dict.items():
        k_lower = k.lower()
        if any(sec in k_lower for sec in SENSITIVE_KEYS):
            cleaned[k] = "[REDACTED]"
        elif isinstance(v, dict):
            cleaned[k] = clean_record_extra(v)
        else:
            cleaned[k] = v
    return cleaned


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
        subset["req"] = clean_record_extra(record["extra"]["req"])
    if "res" in record["extra"]:
        subset["res"] = clean_record_extra(record["extra"]["res"])
    if "responseTime" in record["extra"]:
        subset["responseTime"] = record["extra"]["responseTime"]

    # Include any other extra fields
    exclude_keys = {
        "service",
        "environment",
        "context",
        "traceId",
        "req",
        "res",
        "responseTime",
        "logger_name",
    }
    for key, value in record["extra"].items():
        if key not in exclude_keys:
            key_lower = key.lower()
            if any(sec in key_lower for sec in SENSITIVE_KEYS):
                subset[key] = "******"
            elif isinstance(value, dict):
                subset[key] = clean_record_extra(value)
            else:
                subset[key] = value

    return json.dumps(subset, default=str)


class InterceptHandler(logging.Handler):
    """
    Intercept standard logging messages and redirect them to Loguru.
    """

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


# Create a single global InterceptHandler instance
intercept_handler = InterceptHandler()


def setup_app_logging(log_level: str = "INFO"):
    """
    Setup loguru to handle all application logs.
    Cloud-native design:
      - Dev: Human-readable logs to sys.stderr (colored).
      - Prod: Structured JSON logs directly to sys.stderr (for Docker collection).
    """
    # 0. Suppress common library noise
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*ccache.*")

    # Remove all default handlers
    logger.remove()

    settings = get_settings()

    # 1. Configure Cloud-Native Handlers based on environment
    if settings.app_env == "development":
        # Development Handler: Colored, human-readable console output
        logger.add(
            sys.stderr,
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        )
    else:
        # Production Handler: Single source JSON output straight to sys.stderr
        logger.add(
            sys.stderr,
            level=log_level,
            format="{extra[serialized]}",
        )

        # Bind default context and attach custom serializer only in non-development environments
        logger.configure(
            patcher=lambda record: record["extra"].update(serialized=serialize(record))
        )

    logging.basicConfig(handlers=[intercept_handler], level=0, force=True)

    intercept_only = ["fastapi", "uvicorn", "uvicorn.error"]

    silenced_loggers = ["ppocr", "paddle", "paddlex", "watchfiles"]

    for name in intercept_only + silenced_loggers:
        _logger = logging.getLogger(name)
        _logger.handlers = [intercept_handler]
        _logger.propagate = False

    for name in silenced_loggers:
        logging.getLogger(name).setLevel(logging.ERROR)

    # Disable uvicorn.access logger completely to prevent duplicate access logs
    logging.getLogger("uvicorn.access").disabled = True
