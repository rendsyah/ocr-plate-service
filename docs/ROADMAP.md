# Implementation Roadmap

## Phase 1: Core Engine (Foundations)
- [x] **1. Config & Environment**: Setup `pydantic-settings` for all variables.
- [x] **2. Logging & Observability**: Setup `loguru` with rotation and performance profiling.
- [x] **3. Detector Wrapper**: Interface and YOLOv8 implementation.
- [x] **4. OCR Wrapper**: Interface and PaddleOCR (v3.7+) implementation.
- [x] **5. Basic Pipeline**: Combine Detection + OCR + Indonesian Regex.
- [x] **6. Local Validation**: Test suite using a folder of local images.

## Phase 2: Camera Integration (Completed)
- [x] **7. Snapshot Ingestor**: Implement HTTP/OpenCV logic to fetch images from camera URL.
- [x] **8. Camera Resilience**: Add retry logic and health checks for camera connectivity.
- [x] **9. Pipeline Integration**: Connect the snapshot ingestor with the inference pipeline.

## Phase 3: API & Integration (Current)
- [x] **10. FastAPI Layer**: Implement the trigger endpoint and status checks.
- [x] **11. Snapshot Management**: Logic to save and serve audit images.
- [ ] **12. Backend Core Integration**: Final testing with the Go service.

## Phase 4: Edge Optimization (Deferred - Needs Orange Pi)
- [ ] **13. Model Quantization**: Export to ONNX or RKNN (if using RK3588).
- [ ] **14. Resource Tuning**: Optimize CPU/Memory usage for 24/7 operation.
- [ ] **15. Deployment**: Dockerization and service management (systemd/docker-compose).
