# Product Requirements Document (PRD) - OCR Plate Service

## 1. Objective
Build a production-grade, reusable OCR Plate Service optimized for edge computing (Orange Pi) to be used in various parking system projects.

## 2. Key Requirements
- **High Accuracy**: Must correctly identify Indonesian license plates in various conditions (day, night, tilt, motion blur).
- **Low Latency**: End-to-end processing (from trigger to response) should be < 1.5s on edge hardware.
- **Robustness**: 
    - Handle IP camera snapshot retrieval with retries.
    - Provide confidence scores and "manual review" flags.
    - Graceful failure handling (e.g., Camera Down, AI Model Error).
- **Reusability**: 
    - Stateless API design.
    - Environment-based configuration.
    - Pluggable AI models (YOLO, OCR engines).

## 3. Functional Requirements
- **Snapshot Capture**: Ability to pull a high-resolution snapshot from an IP camera URL upon trigger.
- **Vehicle Detection**: Identify vehicle type (Car, Motorcycle).
- **Plate Recognition**: Extract text from Indonesian license plates.
- **Post-processing**:
    - Indonesian-specific character correction (e.g., `0` vs `O`).
    - Regex validation for Indonesian plate formats.
- **Storage**: Save a snapshot of the processed frame for audit logs.

## 4. Success Metrics
- **OCR Accuracy**: > 95% for clear daytime images, > 85% for nighttime/challenging images.
- **Detection Rate**: > 99% for vehicles passing the loop detector.
- **Stability**: 0% memory leaks over 30 days of continuous operation.
