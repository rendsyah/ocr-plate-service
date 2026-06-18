# Architecture & System Design

## 1. System Overview
The service operates as a standalone microservice at the edge (Orange Pi), sitting between the IP Camera and the Backend Core (Go).

```text
[ IP Camera ] <---(Snapshot URL)--- [ OCR Service (Python/FastAPI) ]
                                            |
[ Backend Core (Go) ] <---(HTTP)-----------[ API Layer ]
```

## 2. Component Breakdown

### A. Snapshot Ingestor
- **Tech**: HTTP Client (httpx) / OpenCV.
- **Role**: Fetches a static image from the IP Camera's snapshot URL upon request.
- **Resilience**: Implements retries and timeouts for camera connectivity.

### B. Inference Pipeline
1. **Trigger**: Receives HTTP GET/POST from Backend Core.
2. **Snapshot Retrieval**: Pulls the latest frame from the camera.
3. **Stage 1: Detector (YOLO)**:
    - Runs inference on the captured frame.
    - Identifies Vehicle Type and Plate Bounding Box.
4. **Stage 2: OCR**:
    - Crops the plate area from the frame.
    - Runs text recognition.
5. **Stage 3: Post-Processor**:
    - Applies `Indonesian Heuristics` (Position-based character mapping).
    - Regex validation.

### C. Storage Provider
- Saves the processed frame as a JPG file for audit purposes.
- Managed via configurable retention.

## 3. Data Flow
1. `GET /status`: Health check for camera and models.
2. `POST /v1/ocr/predict`:
    - Response (OCRData):
      ```json
      {
        "plate_number": "B1234XYZ",
        "vehicle_type": "car",
        "confidence": 0.95,
        "ocr_conf": 0.94,
        "detection_conf": 0.96,
        "is_valid": true,
        "sample_used": null,
        "metadata": {
          "metrics": {
            "detection_ms": 150.0,
            "preprocess_ms": 20.0,
            "ocr_ms": 300.0,
            "normalize_ms": 5.0,
            "total_ms": 475.0
          }
        }
      }
      ```

## 4. Technology Stack
- **Language**: Python 3.12+
- **Framework**: FastAPI (Async)
- **AI/ML**: Ultralytics (YOLOv8/11), PaddleOCR (v3.7+)
- **Ops**: Loguru, Pydantic Settings, Tenacity (Retry logic).
