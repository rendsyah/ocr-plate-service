# Architecture & System Design

## 1. System Overview
The service operates as a standalone OCR microservice at the edge, processing uploaded images via HTTP API.

```text
[ Client / Backend Core ] <---(HTTP POST /api/v1/ocr/predict)--- [ OCR Service (Python/FastAPI) ]
                                                                          |
                                                                     [ Inference Pipeline ]
```

## 2. Component Breakdown

### A. API Layer
- **Tech**: FastAPI (Async).
- **Role**: Accepts image uploads, orchestrates the pipeline, returns OCR results.
- **Endpoints**: Health check, OCR prediction, sample-based prediction.

### B. Inference Pipeline
1. **Trigger**: Receives HTTP POST with uploaded image.
2. **Stage 1: Detector (YOLO)**:
    - Runs inference on the input image.
    - Identifies Vehicle Type and Plate Bounding Box.
3. **Stage 2: OCR**:
    - Crops the plate area from the frame.
    - Runs text recognition via PaddleOCR.
4. **Stage 3: Post-Processor**:
    - Applies `Indonesian Heuristics` (Position-based character mapping).
    - Regex validation and normalization.

### C. Storage Provider
- Saves the processed frame as a JPG file for audit purposes.
- Managed via configurable retention.

## 3. Data Flow
1. `GET /api/v1/health`: Health check for pipeline and storage.
2. `POST /api/v1/ocr/predict`:
    - Request: multipart image upload.
    - Response (OCRData):
      ```json
      {
        "plate_number": "B1234XYZ",
        "vehicle_type": "car",
        "confidence": 0.95,
        "ocr_conf": 0.94,
        "detection_conf": 0.96,
        "is_valid": true,
        "snapshot_filename": "143022_B1234XYZ_0.95.jpg",
        "preprocess_filename": "proc_143022.jpg",
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
- **Ops**: Loguru, Pydantic Settings.
