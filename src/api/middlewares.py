import time
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from src.utils import logger


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to capture Request ID (traceId) and HTTP metadata
    in a structured JSON format for production monitoring.
    """

    async def dispatch(self, request: Request, call_next):
        # Skip logging for health checks
        if request.url.path == "/api/v1/health":
            return await call_next(request)

        start_time = time.time()
        trace_id = str(uuid.uuid4())

        # Bind traceId to all logs in this request context, but NOT context="HTTP"
        # This prevents internal application logs from being tagged as HTTP context.
        with logger.contextualize(traceId=trace_id):
            try:
                response = await call_next(request)
                response_time = round((time.time() - start_time) * 1000, 2)

                # Filter headers to match allowed production fields
                allowed_headers = {
                    "host",
                    "referer",
                    "accept",
                    "user-agent",
                    "content-type",
                    "x-forwarded-for",
                }
                filtered_headers = {
                    k: v
                    for k, v in request.headers.items()
                    if k.lower() in allowed_headers
                }

                # Prepare structured log metadata
                # fmt: off
                log_data = {
                    "req": {
                        "method": request.method,
                        "protocol": request.url.scheme,
                        "httpVersion": request.scope.get("http_version"),
                        "url": str(request.url.path),
                        "remoteAddress": request.client.host if request.client else None,
                        "headers": filtered_headers,
                        "body": {},  # Body logging is skipped for performance/security
                        "query": dict(request.query_params),
                        "params": dict(request.path_params),
                    },
                    "res": {"statusCode": response.status_code},
                    "error": None,
                    "responseTime": response_time,
                    "traceId": trace_id,
                    "logger_name": "uvicorn.access",  # Routes to http.log
                }

                # Explicitly bind context="HTTP" ONLY for this access log
                logger.bind(context="HTTP", **log_data).info(
                    f"{request.method} {request.url.path}"
                )

                return response
            except Exception as e:
                response_time = round((time.time() - start_time) * 1000, 2)
                # fmt: off
                log_data = {
                    "req": {
                        "method": request.method,
                        "protocol": request.url.scheme,
                        "httpVersion": request.scope.get("http_version"),
                        "url": str(request.url.path),
                        "remoteAddress": request.client.host if request.client else None,
                        "headers": dict(request.headers),
                        "body": {},
                        "query": dict(request.query_params),
                        "params": dict(request.path_params),
                    },
                    "res": None,
                    "error": str(e),
                    "responseTime": response_time,
                    "traceId": trace_id,
                    "logger_name": "uvicorn.access",
                }
                logger.bind(context="HTTP", **log_data).error(
                    f"{request.method} {request.url.path} - Error: {e}"
                )
                raise e
