# OCR Plate Service

A production-grade, reusable OCR Plate Service for parking system projects. This service detects vehicles and recognizes Indonesian license plates from uploaded images.

## 🚀 Features

- **High Accuracy**: Optimized YOLOv8/v11 for vehicle/plate detection.
- **PaddleOCR Powered**: Optimized PaddleOCR (v3.7+) for license plate recognition.
- **FastAPI Powered**: Async API layer with built-in validation and performance metrics.
- **Clean Architecture**: Decoupled domain, services, and API layers for easy maintenance.
- **Indonesian Format**: Specialized normalization and regex validation for Indonesian license plates.

## 🛠️ Tech Stack

- **Language**: Python 3.12+
- **Framework**: FastAPI
- **AI/ML**: Ultralytics (YOLO), PaddleOCR
- **Tools**: UV (Package Management), OpenCV, Pydantic Settings, Loguru

## 📋 Prerequisites

- Python 3.12+
- `uv` installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

## ⚙️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ocr-plate-service
   ```

2. **Setup environment and hooks**:
   ```bash
   make install
   ```
   *This will install dependencies via `uv` and setup git hooks for commit messages and pre-push checks.*

3. **Configure Environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your model paths
   ```

4. **Prepare Models**:
   Ensure your YOLO model is placed in the `models/` directory (default: `models/license_plate.pt`).

## 🛠️ Development

This project uses a `Makefile` to automate common tasks:

- `make lint`      : Run code quality checks (Ruff).
- `make format`    : Auto-format code.
- `make validate`  : Run end-to-end OCR validation on sample images.
- `make test`      : Run unit tests.
- `make clean`     : Remove temporary caches and logs.

## 🏃 Running the Application

Using Makefile:
```bash
make run
```

Or directly using uv:
```bash
uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8080
```

### 🐳 Running with Docker

For production deployment, it is recommended to use Docker:

1. **Build and start**:
   ```bash
   docker-compose up -d --build
   ```

2. **Check logs**:
   ```bash
   docker-compose logs -f
   ```

3. **Stop the service**:
   ```bash
   docker-compose down
   ```

## 🧪 API Usage

Once the server is running, visit `http://localhost:8080/docs` for the interactive Swagger UI.

### Key Endpoints:

- `GET /api/v1/health`: Check system status.
- `POST /api/v1/ocr/predict`: Upload an image file for OCR prediction.
- `POST /api/v1/ocr/predict-test`: Process a random sample image from `tests/samples/`.

## 📁 Project Structure

- `src/api`: FastAPI routes, schemas, and dependencies.
- `src/config`: App settings and OpenAPI metadata.
- `src/domain`: Base interfaces and domain logic.
- `src/services`: Core logic (Pipeline, Storage, Detector, OCR).
- `storage/`: Local storage for snapshots and debug images.
- `tests/`: Unit tests and sample images.

## 📄 License

Apache 2.0
