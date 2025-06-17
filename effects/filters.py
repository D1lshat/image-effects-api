import cv2
import numpy as np


def apply_warm_cold_filter(image: np.ndarray, warmth: float = 1.2, coldness: float = 0.9) -> np.ndarray:
    """Увеличивает красные оттенки (тепло) и уменьшает синие (холодно)."""
    b, g, r = cv2.split(image.astype('float32'))
    r = np.clip(r * warmth, 0, 255).astype(np.uint8)
    b = np.clip(b * coldness, 0, 255).astype(np.uint8)
    g = g.astype(np.uint8)  # Приводим к тому же типу
    return cv2.merge([b, g, r])


def apply_negative(image: np.ndarray) -> np.ndarray:
    """Превращает изображение в негативное."""
    return cv2.bitwise_not(image)


def apply_tint(image: np.ndarray, r: int = 0, g: int = 0, b: int = 50) -> np.ndarray:
    """Накладывает цветовой фильтр (RGB)."""
    tint = np.full_like(image, (b, g, r), dtype=np.uint8)
    return cv2.addWeighted(image, 0.8, tint, 0.2, 0)


def solarize(image: np.ndarray, threshold: int = 128) -> np.ndarray:
    """Эффект соляризации — частичная инверсия яркости."""
    return cv2.threshold(image, threshold, 255, cv2.THRESH_TOZERO)[1]


def black_white_negative(image: np.ndarray) -> np.ndarray:
    """Черно-белое негативное изображение."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.bitwise_not(gray)