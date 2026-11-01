from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError as PydanticValidationError

from core.logging import get_logger

logger = get_logger(__name__)


class AppException(Exception):
    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code: str = "INTERNAL_ERROR",
        details: dict | None = None,
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}
        super().__init__(message)


class ConfigurationError(AppException):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="CONFIGURATION_ERROR",
            details=details,
        )


class NotFoundError(AppException):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="NOT_FOUND",
            details=details,
        )


class AppValidationError(AppException):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code="VALIDATION_ERROR",
            details=details,
        )


def create_error_response(
    request: Request,
    status_code: int,
    error_code: str,
    message: str,
    details: dict | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": error_code,
                "message": message,
                "details": details or {},
                "path": str(request.url.path),
            }
        },
    )


async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    logger.warning(
        "application_exception",
        error_code=exc.error_code,
        message=exc.message,
        details=exc.details,
        path=str(request.url.path),
    )
    return create_error_response(
        request,
        exc.status_code,
        exc.error_code,
        exc.message,
        exc.details,
    )


def _sanitize_errors(errors: list) -> list:
    sanitized = []
    for err in errors:
        e = dict(err)
        if "ctx" in e and isinstance(e["ctx"], dict):
            e["ctx"] = {k: str(v) for k, v in e["ctx"].items()}
        sanitized.append(e)
    return sanitized


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    clean_errors = _sanitize_errors(exc.errors())
    logger.warning(
        "validation_error",
        errors=clean_errors,
        path=str(request.url.path),
    )
    return create_error_response(
        request,
        status.HTTP_422_UNPROCESSABLE_ENTITY,
        "VALIDATION_ERROR",
        "Request validation failed",
        {"errors": clean_errors},
    )


async def pydantic_validation_exception_handler(request: Request, exc: PydanticValidationError) -> JSONResponse:
    clean_errors = _sanitize_errors(exc.errors())
    logger.warning(
        "pydantic_validation_error",
        errors=clean_errors,
        path=str(request.url.path),
    )
    return create_error_response(
        request,
        status.HTTP_422_UNPROCESSABLE_ENTITY,
        "VALIDATION_ERROR",
        "Data validation failed",
        {"errors": clean_errors},
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    settings = request.app.state.settings if hasattr(request.app.state, "settings") else None
    is_debug = settings.debug if settings else False

    logger.error(
        "unhandled_exception",
        error_type=type(exc).__name__,
        message=str(exc),
        path=str(request.url.path),
        exc_info=True,
    )

    if is_debug:
        return create_error_response(
            request,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "INTERNAL_ERROR",
            str(exc),
            {"type": type(exc).__name__},
        )

    return create_error_response(
        request,
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "INTERNAL_ERROR",
        "An unexpected error occurred",
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppException, app_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(PydanticValidationError, pydantic_validation_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)