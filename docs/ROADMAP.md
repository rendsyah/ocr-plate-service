# Implementation Roadmap

## Phase 1: Core Engine (Foundations)
- [x] **1. Config & Environment**: Setup `pydantic-settings` for all variables.
- [x] **2. Logging & Observability**: Setup `loguru` with structured JSON logging.
- [x] **3. Detector Wrapper**: Interface and YOLO implementation.
- [x] **4. OCR Wrapper**: Interface and PaddleOCR implementation.
- [x] **5. Basic Pipeline**: Combine Detection + OCR + Indonesian Regex.
- [x] **6. Local Validation**: Test suite using a folder of local images.

## Phase 2: API & Integration
- [x] **7. FastAPI Layer**: Implement prediction endpoints and health checks.
- [x] **8. Snapshot Management**: Logic to save and serve audit images.
- [x] **9. Production Hardening**: Structured logging, error handling, middleware.
