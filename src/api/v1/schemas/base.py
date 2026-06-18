import uuid
from typing import Any, Generic, List, Optional, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class SuccessResponse(BaseModel, Generic[T]):
    """Standardized successful API response."""

    success: bool = True
    message: str = "Success"
    data: T


class ErrorResponse(BaseModel):
    """Standardized error API response."""

    success: bool = False
    message: str
    data: Optional[Any] = None
    errors: List[Any] = []
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
