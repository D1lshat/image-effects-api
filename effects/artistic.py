import cv2
import numpy as np


def pencil_sketch(image: np.ndarray) -> np.ndarray:
    """Превращает изображение в карандашный набросок."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inv = cv2.bitwise_not(gray)
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)


def apply_watercolor(image: np.ndarray) -> np.ndarray:
    """Применяет эффект акварели с помощью stylization."""
    return cv2.stylization(image, sigma_s=60, sigma_r=0.6)


def oil_painting_effect(image: np.ndarray) -> np.ndarray:
    """Эффект масляной картины (требуется opencv-contrib-python)."""
    try:
        return cv2.xphoto.oilPainting(image, size=7, dynRatio=1)
    except AttributeError:
        raise ImportError("cv2.xphoto module not found. "
                          "Install 'opencv-contrib-python' package.")


def pencil_with_color_effect(image: np.ndarray) -> np.ndarray:
    """Карандашный контур поверх цветного фона."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted_gray = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(inverted_gray, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blurred, scale=256)

    color = cv2.bilateralFilter(image, d=9, sigmaColor=300, sigmaSpace=300)

    result = np.zeros_like(image)
    for c in range(3):
        result[:, :, c] = cv2.multiply(sketch, color[:, :, c], scale=1 / 256.0)

    return result


def cartoonify(image: np.ndarray) -> np.ndarray:
    """Превращает изображение в мультяшное."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(blur, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)

    color = cv2.bilateralFilter(image, d=9, sigmaColor=300, sigmaSpace=300)
    return cv2.bitwise_and(color, color, mask=edges)