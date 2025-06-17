from fastapi import Depends, FastAPI, UploadFile, File, Form, Query, HTTPException, Response,APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from enum import Enum
import numpy as np
import cv2
from pydantic import BaseModel
from typing import List, Dict, Any,Union
import json
import io
import requests
import uvicorn  
from openai import OpenAI
# Импорты эффектов
from effects import utils
from effects import stylization
from effects.image_routes import router as  gemini_image_router
from effects.basic import (
    gaussian_blur,
    median_blur,
    average_blur,
    bilateral_blur,
    grayscale,
    invert_colors,
    adjust_brightness,
    adjust_contrast
)
from effects.artistic import (
    pencil_sketch,
    apply_watercolor,
    oil_painting_effect,
    pencil_with_color_effect,
    cartoonify
)
from effects.filters import (
    apply_warm_cold_filter,
    apply_negative,
    apply_tint,
    solarize,
    black_white_negative
)
from effects.stylization import (
    style_transfer,
    apply_hdr_effect,
    pixelate,
    vignette_effect,
    film_grain_effect,
    im_convert,
    
)
from effects.background import (
    remove_background,
    blur_background 
)
from effects.edges import (
    canny_edge_detection,
    sobel_edge_detection,
    laplacian_edge_detection,
    scharr_edge_detection,
    simple_contour_detection
)
from effects.enhancements import (
    sharpen_image,
    reduce_noise,
    enhance_details,
    equalize_exposure,
    apply_clahe_filter
)
from effects.utils import (
    load_image,
    image_to_bytes,
    validate_image,
    normalize_image,
    image_to_bytes
    
)

app = FastAPI()

# Импортируем реестр фильтров
from effects.effect_registry import EFFECT_REGISTRY
class FilterRequest(BaseModel):
    name: str
    params: Dict[str, Any] = {}

class EffectParamModel(BaseModel):
    name: str
    params: Dict[str, Union[int, float, str, bool, None]] = {}


class ImageProcessRequest(BaseModel):
    filters: List[EffectParamModel]
    
class ProcessRequest(BaseModel):
    filters: List[FilterRequest]



# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class BlurType(str, Enum):
    gaussian = "gaussian"
    average = "average"
    median = "median"
    bilateral = "bilateral"

@app.post("/apply/")
async def apply_filters(
    file: UploadFile = File(...),
    filters: str = Form(...)  # Получаем как строку
):
    try:
        # Загружаем изображение
        image = await load_image(file)
        image = validate_image(image)

        # Парсим JSON строку в список словарей
        try:
            filters_list = json.loads(filters)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid filters JSON")

        if not isinstance(filters_list, list):
            raise HTTPException(status_code=400, detail="Filters must be a list of effects")

        result = image.copy()

        for filter_info in filters_list:
            effect_name = filter_info.get("name")
            params = filter_info.get("params", {})

            if effect_name not in EFFECT_REGISTRY:
                raise HTTPException(status_code=400, detail=f"Unknown effect '{effect_name}'")

            effect_func = EFFECT_REGISTRY[effect_name]
            result = effect_func(result, **params)

        final_result = normalize_image(result)
        return Response(content=image_to_bytes(final_result, ".jpg"), media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
@app.post("/process/", summary="Apply multiple effects in sequence", description="Apply a chain of effects to an image.")
async def process_image(
    file: UploadFile = File(..., description="Image file to apply effects on"),
    filters: List[FilterRequest] = Depends(),
):
    try:
        # Загрузка и валидация изображения
        image = await load_image(file)
        image = validate_image(image)

        # Последовательное применение фильтров
        result = image.copy()
        for filter_info in filters:
            effect_name = filter_info.name
            params = filter_info.params or {}

            if effect_name not in EFFECT_REGISTRY:
                raise HTTPException(status_code=400, detail=f"Unknown effect: {effect_name}")

            effect_func = EFFECT_REGISTRY[effect_name]
            result = effect_func(result, **params)

        final_result = normalize_image(result)
        return Response(content=image_to_bytes(final_result, ".jpg"), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    

# ================== Базовые эффекты ==================
@app.post("/blur/")
async def apply_blur(
    file: UploadFile = File(...),
    blur_type: BlurType = Form(BlurType.gaussian),
    kernel_size: int = Form(15),
    diameter: int = Form(9),
    sigma_color: int = Form(75),
    sigma_space: int = Form(75)
):
    try:
        image = await load_image(file)
        image = validate_image(image)

        if blur_type == BlurType.gaussian:
            result = gaussian_blur(image, kernel_size)
        elif blur_type == BlurType.average:
            result = average_blur(image, kernel_size)
        elif blur_type == BlurType.median:
            result = median_blur(image, kernel_size)
        elif blur_type == BlurType.bilateral:
            result = bilateral_blur(image, diameter, sigma_color, sigma_space)
        else:
            raise HTTPException(status_code=400, detail="Invalid blur type")

        return Response(content=image_to_bytes(result, ".jpg"), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Blur effect failed: {str(e)}")


@app.post("/grayscale/")
async def apply_grayscale(file: UploadFile = File(...)):
    try:
        image = await load_image(file)
        image = validate_image(image)
        result = grayscale(image)
        return Response(content=image_to_bytes(result, ".jpg"), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grayscale failed: {str(e)}")


@app.post("/invert_colors/")
async def apply_invert(file: UploadFile = File(...)):
    try:
        image = await load_image(file)
        image = validate_image(image)
        result = invert_colors(image)
        return Response(content=image_to_bytes(result, ".jpg"), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Invert colors failed: {str(e)}")


@app.post("/brightness/")
async def change_brightness(file: UploadFile = File(...), brightness: int = Query(30)):
    try:
        image = await load_image(file)
        image = validate_image(image)
        result = adjust_brightness(image, brightness)
        return Response(content=image_to_bytes(result, ".jpg"), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Brightness adjustment failed: {str(e)}")


@app.post("/contrast/")
async def change_contrast(file: UploadFile = File(...), contrast: float = Query(1.0)):
    try:
        image = await load_image(file)
        image = validate_image(image)
        result = adjust_contrast(image, contrast)
        return Response(content=image_to_bytes(result, ".jpg"), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Contrast adjustment failed: {str(e)}")


# ================== Художественные эффекты ==================
@app.post("/pencil/")
async def apply_pencil_sketch(file: UploadFile = File(...)):
    try:
        image = await load_image(file)
        image = validate_image(image)
        result = pencil_sketch(image)
        return Response(content=image_to_bytes(result, ".jpg"), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pencil sketch failed: {str(e)}")


@app.post("/watercolor/")
async def apply_watercolor_effect(file: UploadFile = File(...)):
    try:
        image = await load_image(file)
        image = validate_image(image)
        result = apply_watercolor(image)
        return Response(content=image_to_bytes(result, ".jpg"), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Watercolor effect failed: {str(e)}")


@app.post("/oil_painting/")
async def apply_oil_painting(file: UploadFile = File(...)):
    try:
        image = await load_image(file)
        image = validate_image(image)
        result = oil_painting_effect(image)
        return Response(content=image_to_bytes(result, ".jpg"), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Oil painting failed: {str(e)}")


@app.post("/pencil_color/")
async def apply_pencil_color(file: UploadFile = File(...)):
    try:
        image = await load_image(file)
        image = validate_image(image)
        result = pencil_with_color_effect(image)
        return Response(content=image_to_bytes(result, ".jpg"), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Color pencil failed: {str(e)}")


@app.post("/cartoonify/")
async def apply_cartoonify(file: UploadFile = File(...)):
    try:
        image = await load_image(file)
        image = validate_image(image)
        result = cartoonify(image)
        return Response(content=image_to_bytes(result, ".jpg"), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cartoonify failed: {str(e)}")


# ================== Цветовые фильтры ==================
@app.post("/warm_cold/")
async def apply_warm_cold(
    file: UploadFile = File(...),
    warmth: float = Query(1.2),
    coldness: float = Query(0.9)
):
    try:
        image = await load_image(file)
        image = validate_image(image)
        result = apply_warm_cold_filter(image, warmth, coldness)
        return Response(content=image_to_bytes(result, ".jpg"), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Warm/Cold filter failed: {str(e)}")


@app.post("/negative/")
async def apply_negative_filter(file: UploadFile = File(...)):
    try:
        image = await load_image(file)
        image = validate_image(image)
        result = apply_negative(image)
        return Response(content=image_to_bytes(result, ".jpg"), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Negative filter failed: {str(e)}")


@app.post("/tint/")
async def apply_tint_filter(
    file: UploadFile = File(...),
    r: int = Query(0),
    g: int = Query(0),
    b: int = Query(50)
):
    try:
        image = await load_image(file)
        image = validate_image(image)
        result = apply_tint(image, r, g, b)
        return Response(content=image_to_bytes(result, ".jpg"), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tint filter failed: {str(e)}")


@app.post("/solarize/")
async def apply_solarize(file: UploadFile = File(...), threshold: int = Query(128)):
    try:
        image = await load_image(file)
        image = validate_image(image)
        result = solarize(image, threshold)
        return Response(content=image_to_bytes(result, ".jpg"), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Solarize effect failed: {str(e)}")


@app.post("/bw_negative/")
async def apply_bw_negative(file: UploadFile = File(...)):
    try:
        image = await load_image(file)
        image = validate_image(image)
        result = black_white_negative(image)
        return Response(content=image_to_bytes(result, ".jpg"), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Black & White negative failed: {str(e)}")


# ================== Эффекты стилизации ==================
@app.post("/style_transfer/")
async def apply_style_transfer(
    content_file: UploadFile = File(...),
    style_file: UploadFile = File(...)
):
    try:
        result_image = style_transfer(content_file, style_file)
        return Response(content=utils.image_to_bytes(result_image, ".jpg"), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Style transfer failed: {str(e)}")
@app.post("/hdr/")
async def apply_hdr(file: UploadFile = File(...)):
    try:
        image = await load_image(file)
        image = validate_image(image)
        result = apply_hdr_effect(image)
        return Response(content=image_to_bytes(result, ".jpg"), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"HDR effect failed: {str(e)}")


@app.post("/pixelate/")
async def apply_pixelate(file: UploadFile = File(...), block_size: int = Query(10)):
    try:
        image = await load_image(file)
        image = validate_image(image)
        result = pixelate(image, block_size)
        return Response(content=image_to_bytes(result, ".jpg"), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pixelate effect failed: {str(e)}")


@app.post("/vignette/")
async def apply_vignette(file: UploadFile = File(...), strength: float = Query(2.0)):
    try:
        image = await load_image(file)
        image = validate_image(image)
        result = vignette_effect(image, strength)
        return Response(content=image_to_bytes(result, ".jpg"), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vignette effect failed: {str(e)}")


@app.post("/film_grain/")
async def apply_film_grain(file: UploadFile = File(...), intensity: float = Query(0.05)):
    try:
        image = await load_image(file)
        image = validate_image(image)
        result = film_grain_effect(image, intensity)
        return Response(content=image_to_bytes(result, ".jpg"), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Film grain effect failed: {str(e)}")


# ================== Работа с фоном ==================
@app.post("/remove_background/")
async def apply_remove_background(file: UploadFile = File(...)):
    try:
        image = await load_image(file)
        image = validate_image(image)

        result = remove_background(image)
        return Response(content=image_to_bytes(result, ".png"), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Background removal failed: {str(e)}")
@app.post("/blur_background/")
async def apply_blur_background(
    file: UploadFile = File(...),
    blur_intensity: int = Query(51, ge=3, le=99)
):
    try:
        image = await load_image(file)
        image = validate_image(image)

        result = blur_background(image, blur_intensity)
        return Response(content=image_to_bytes(result, ".jpg"), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Background blur failed: {str(e)}")
# ================== Контурные / краевые эффекты ==================
@app.post("/canny/")
async def apply_canny(file: UploadFile = File(...), threshold1: int = Query(100), threshold2: int = Query(200)):
    try:
        image = await load_image(file)
        image = validate_image(image)
        result = canny_edge_detection(image, threshold1, threshold2)
        return Response(content=image_to_bytes(result, ".jpg"), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Canny edge detection failed: {str(e)}")


@app.post("/sobel/")
async def apply_sobel(file: UploadFile = File(...), kernel_size: int = Query(3, ge=1, le=7)):
    try:
        image = await load_image(file)
        image = validate_image(image)
        result = sobel_edge_detection(image, kernel_size)
        return Response(content=image_to_bytes(result, ".jpg"), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sobel edge detection failed: {str(e)}")


@app.post("/laplacian/")
async def apply_laplacian(file: UploadFile = File(...)):
    try:
        image = await load_image(file)
        image = validate_image(image)
        result = laplacian_edge_detection(image)
        return Response(content=image_to_bytes(result, ".jpg"), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Laplacian edge detection failed: {str(e)}")


@app.post("/scharr/")
async def apply_scharr(file: UploadFile = File(...)):
    try:
        image = await load_image(file)
        image = validate_image(image)
        result = scharr_edge_detection(image)
        return Response(content=image_to_bytes(result, ".jpg"), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scharr edge detection failed: {str(e)}")


@app.post("/contours/")
async def apply_contours(file: UploadFile = File(...)):
    try:
        image = await load_image(file)
        image = validate_image(image)
        result = simple_contour_detection(image)
        return Response(content=image_to_bytes(result, ".jpg"), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Contour detection failed: {str(e)}")


# ================== Улучшение качества изображения ==================
@app.post("/sharpen/")
async def apply_sharpen(file: UploadFile = File(...), strength: float = Query(1.5)):
    try:
        image = await load_image(file)
        image = validate_image(image)
        result = sharpen_image(image, strength)
        return Response(content=image_to_bytes(result, ".jpg"), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sharpening failed: {str(e)}")


@app.post("/reduce_noise/")
async def apply_reduce_noise(file: UploadFile = File(...), noise_strength: int = Query(10)):
    try:
        image = await load_image(file)
        image = validate_image(image)
        result = reduce_noise(image, h=noise_strength)
        return Response(content=image_to_bytes(result, ".jpg"), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Noise reduction failed: {str(e)}")


@app.post("/enhance_details/")
async def apply_enhance_details(file: UploadFile = File(...)):
    try:
        image = await load_image(file)
        image = validate_image(image)
        result = enhance_details(image)
        return Response(content=image_to_bytes(result, ".jpg"), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detail enhancement failed: {str(e)}")


@app.post("/equalize_exposure/")
async def apply_equalize_exposure(file: UploadFile = File(...)):
    try:
        image = await load_image(file)
        image = validate_image(image)
        result = equalize_exposure(image)
        return Response(content=image_to_bytes(result, ".jpg"), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Exposure equalization failed: {str(e)}")

@app.post("/clahe/")
async def apply_clahe(file: UploadFile = File(...), clip_limit: float = Query(2.0)):
    try:
        # ✔️ Это правильно: чтение через await
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # ✔️ Передаём только numpy-изображение
        result = apply_clahe_filter(image, clip_limit=clip_limit)

        _, encoded = cv2.imencode('.jpg', result)
        return Response(content=encoded.tobytes(), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CLAHE failed: {str(e)}")
    

# ================== генерация изображения ==================

app.include_router(gemini_image_router)