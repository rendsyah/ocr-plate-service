from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.responses import JSONResponse

from src.api.v1.schemas import ErrorResponse
from src.utils import logger


def setup_exception_handlers(app: FastAPI):
    """
    Register all custom exception handlers to the FastAPI app.
    """

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Global handler for HTTPExceptions to return ErrorResponse format."""
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(message=exc.detail, errors=[]).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        """Handler for Pydantic validation errors (422) to keep format consistent."""
        errors = [
            f"{'.'.join(map(str, error['loc']))}: {error['msg']}"
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                message="Validation Error", errors=errors
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Global handler for unhandled exceptions."""
        logger.bind(context="EXCEPTION").exception(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                message="Internal Server Error", errors=[]
            ).model_dump(),
        )
