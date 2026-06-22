"""OpenAPI metadata and schema customization for Swagger UI."""

from typing import Any, Dict

from fastapi import FastAPI

APP_TITLE = "Official OCR Plate Service"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = """
The OCR Plate Service API provides a comprehensive interface for:
* **License Plate Detection**: Identifying vehicle plates in images.
* **OCR Recognition**: Extracting text from detected plates using AI engines.
* **Validation**: Ensuring extracted plates match country-specific formats (e.g., Indonesia).

This documentation is intended for internal development teams and authorized clients.
"""


def setup_custom_openapi(app: FastAPI) -> None:
    """
    Override default 4xx/5xx responses in Swagger to use ErrorResponse format
    instead of FastAPI's default HTTPValidationError schema.
    """
    from src.api.v1.schemas import ErrorDetail, ErrorResponse

    _base_openapi = app.openapi  # type: ignore[attr-defined]

    def _custom_openapi() -> Dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema

        schema = _base_openapi()

        error_ref = {"$ref": "#/components/schemas/ErrorResponse"}
        for path in schema.get("paths", {}).values():
            for operation in path.values():
                responses = operation.get("responses", {})
                for code in list(responses.keys()):
                    if code != "200" and code.isdigit() and int(code) >= 400:
                        responses[code] = {
                            "description": responses[code].get("description", "Error"),
                            "content": {
                                "application/json": {
                                    "schema": error_ref,
                                }
                            },
                        }

        schemas = schema.setdefault("components", {}).setdefault("schemas", {})
        schemas["ErrorDetail"] = ErrorDetail.model_json_schema()

        error_schema = ErrorResponse.model_json_schema(
            ref_template="#/components/schemas/{model}"
        )
        error_schema.pop("$defs", None)
        schemas["ErrorResponse"] = error_schema

        app.openapi_schema = schema
        return schema

    app.openapi = _custom_openapi  # type: ignore[attr-defined]
