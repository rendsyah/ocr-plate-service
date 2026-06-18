from fastapi import APIRouter, Depends, File, UploadFile

from src.api.deps import get_ocr_api_service
from src.api.v1.schemas import OCRData, SuccessResponse
from src.api.v1.services import OCRApiService

router = APIRouter()


@router.post("/predict", response_model=SuccessResponse[OCRData])
async def predict(
    service: OCRApiService = Depends(get_ocr_api_service),
) -> SuccessResponse[OCRData]:
    """
    Trigger OCR prediction by capturing a snapshot from the configured camera.
    """
    data = await service.predict()
    return SuccessResponse(data=data)


@router.post("/predict-image", response_model=SuccessResponse[OCRData])
async def predict_image(
    file: UploadFile = File(...),
    service: OCRApiService = Depends(get_ocr_api_service),
) -> SuccessResponse[OCRData]:
    """
    Upload an image file for OCR prediction.
    """
    data = await service.predict_image(file)
    return SuccessResponse(data=data)


@router.post("/predict-test", response_model=SuccessResponse[OCRData])
async def predict_test(
    service: OCRApiService = Depends(get_ocr_api_service),
) -> SuccessResponse[OCRData]:
    """
    Predict using a random sample image.
    """
    data = await service.predict_test()
    return SuccessResponse(data=data)
