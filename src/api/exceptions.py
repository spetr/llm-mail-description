"""Custom API exceptions."""

from fastapi import HTTPException, status


class ServiceUnavailableError(HTTPException):
    """Raised when the inference backend is not available."""

    def __init__(self, detail: str = "Service temporarily unavailable") -> None:
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
        )


class InferenceError(HTTPException):
    """Raised when inference fails."""

    def __init__(self, detail: str = "Inference failed") -> None:
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
        )


class ValidationError(HTTPException):
    """Raised when input validation fails."""

    def __init__(self, detail: str = "Invalid input") -> None:
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail,
        )
