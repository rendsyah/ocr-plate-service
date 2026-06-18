import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.errors import setup_exception_handlers
from src.api.middlewares import LoggingMiddleware
from src.api.v1 import api_router
from src.camera import get_camera
from src.config import APP_DESCRIPTION, APP_TITLE, APP_VERSION, get_settings
from src.services import (
    OCRPipeline,
    get_detector,
    get_normalizer,
    get_ocr,
    get_storage,
)
from src.utils import logger, setup_app_logging

# Initiliaze config settings
settings = get_settings()

# Apply unified logging (includes warning suppression)
setup_app_logging(log_level=settings.log_level)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handle startup and shutdown events.
    Initialize models once during startup and attach to app state.
    """
    try:
        logger.info("Initializing OCR Pipeline...")

        detector = get_detector()
        ocr = get_ocr()
        normalizer = get_normalizer()
        camera = get_camera()
        storage = get_storage()

        app.state.storage = storage
        app.state.camera = camera
        app.state.pipeline = OCRPipeline(detector, ocr, normalizer, storage)

        cleanup_task = asyncio.create_task(storage.run_cleanup_worker())
        app.state.cleanup_task = cleanup_task

        logger.success("OCR Pipeline initialized successfully.")
    except Exception as e:
        logger.critical(f"Failed to initialize OCR Pipeline: {e}")
        raise e

    yield

    if hasattr(app.state, "cleanup_task"):
        app.state.cleanup_task.cancel()
        logger.info("Background cleanup task cancelled.")

    logger.info("Shutting down OCR Service...")


app = FastAPI(
    lifespan=lifespan,
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    contact={
        "name": "Developer",
        "url": "http://www.example.com/support",
        "email": "rndyfrdynsyh@gmail.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
    docs_url="/docs",
    redoc_url="/redoc",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
)

# Register Global Exception Handlers
setup_exception_handlers(app)

# Register Middleware
app.add_middleware(LoggingMiddleware)

# Register Routes With Prefix
app.include_router(api_router, prefix="/api/v1")
