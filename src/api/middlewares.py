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
            k: v for k, v in request.headers.items() if k.lower() in allowed_headers
        }

        # Bind traceId to all logs in this request context
        with logger.contextualize(traceId=trace_id):
            try:
                response = await call_next(request)
                response_time = round((time.time() - start_time) * 1000, 2)

                log_data = self._build_log_data(
                    request, filtered_headers, trace_id, response_time
                )
                log_data["res"] = {"statusCode": response.status_code}

                http_logger = logger.bind(context="HTTP", **log_data)
                msg = f"{request.method} {request.url.path}"

                if response.status_code >= 500:
                    http_logger.error(msg)
                elif response.status_code >= 400:
                    http_logger.warning(msg)
                else:
                    http_logger.info(msg)

                return response
            except Exception as e:
                response_time = round((time.time() - start_time) * 1000, 2)

                log_data = self._build_log_data(
                    request, filtered_headers, trace_id, response_time
                )
                log_data["res"] = None
                log_data["error"] = str(e)

                logger.bind(context="HTTP", **log_data).error(
                    f"{request.method} {request.url.path}"
                )
                raise e

    @staticmethod
    def _build_log_data(request, filtered_headers, trace_id, response_time):
        return {
            "req": {
                "method": request.method,
                "protocol": request.url.scheme,
                "httpVersion": request.scope.get("http_version"),
                "url": str(request.url.path),
                "remoteAddress": request.client.host if request.client else None,
                "headers": filtered_headers,
                "body": {},
                "query": dict(request.query_params),
                "params": dict(request.path_params),
            },
            "responseTime": response_time,
            "traceId": trace_id,
            "logger_name": "http.middleware",
        }
