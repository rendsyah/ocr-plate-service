import uuid
from typing import Generic, List, Optional, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class ErrorDetail(BaseModel):
    """Structured field-level error detail."""

    field: str
    message: str


class SuccessResponse(BaseModel, Generic[T]):
    """Standardized successful API response."""

    success: bool = True
    message: str = "Success"
    data: T


class ErrorResponse(BaseModel):
    """Standardized error API response."""

    success: bool = False
    message: str
    data: Optional[dict] = None
    errors: List[ErrorDetail] = []
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
