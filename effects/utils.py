from PIL import Image
import cv2
from torchvision import transforms
import numpy as np
from fastapi import UploadFile, HTTPException
import torch

async def load_image(file: UploadFile) -> np.ndarray:
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image")
    return image


def image_to_bytes(image: np.ndarray, fmt: str = ".jpg"):
    _, encoded = cv2.imencode(fmt, image)
    return encoded.tobytes()
def validate_image(image: np.ndarray):
    """Проверяет, что изображение корректное (не пустое, правильный тип)."""
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    if not isinstance(image, np.ndarray):
        raise HTTPException(status_code=400, detail="Image must be a numpy array")
    if len(image.shape) not in (2, 3):
        raise HTTPException(status_code=400, detail="Invalid image format or number of channels")
    return image
def validate_image(image: np.ndarray):
    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be a valid numpy array")
    if len(image.shape) < 2:
        raise ValueError("Image must have at least 2 dimensions")
    return image

def convert_to_bgr(image: np.ndarray) -> np.ndarray:
    """Преобразует изображение в BGR, если оно не в этом формате."""
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.shape[2] == 3:
        return image  # Уже BGR или RGB (OpenCV работает с BGR)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported number of channels: {image.shape[2]}")


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Нормализует изображение к диапазону [0..255] и типу uint8."""
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def resize_image(image: np.ndarray, max_size: int = 1024) -> np.ndarray:
    """Уменьшает размер изображения до максимального значения по ширине или высоте."""
    h, w = image.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image


def check_file_extension(filename: str, allowed_extensions: list = ['.jpg', '.jpeg', '.png']):
    """Проверяет расширение файла."""
    ext = filename[filename.rfind('.'):].lower()
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"File extension '{ext}' not supported")
    return ext


def image_to_bytes(image: np.ndarray, fmt: str = ".jpg") -> bytes:
    """Кодирует изображение в байты указанного формата."""
    if fmt.lower() in (".jpg", ".jpeg"):
        _, encoded = cv2.imencode(".jpg", image)
    elif fmt.lower() == ".png":
        _, encoded = cv2.imencode(".png", image)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {fmt}")
    return encoded.tobytes()


def save_image(path: str, image: np.ndarray):
    """Сохраняет изображение на диск."""
    cv2.imwrite(path, image)
    
def smart_resize(image: np.ndarray, max_side: int = 512) -> np.ndarray:
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return image