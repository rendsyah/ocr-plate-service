from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.responses import JSONResponse

from src.api.v1.schemas import ErrorDetail, ErrorResponse
from src.utils import logger


def setup_exception_handlers(app: FastAPI):
    """
    Register all custom exception handlers to the FastAPI app.
    All handlers return a consistent ErrorResponse format.
    """

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Global handler for HTTPExceptions to return ErrorResponse format."""
        errors: list[ErrorDetail] = []
        detail = exc.detail

        if isinstance(detail, list):
            errors = []
            for err in detail:
                if isinstance(err, dict):
                    errors.append(
                        ErrorDetail(
                            field=".".join(map(str, err.get("loc", []))),
                            message=err.get("msg", str(err)),
                        )
                    )
            detail = "Validation Error"

        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(message=detail, errors=errors).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        """Handler for Pydantic validation errors (422) to keep format consistent."""
        errors = [
            ErrorDetail(
                field=".".join(map(str, error["loc"])),
                message=error["msg"],
            )
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
