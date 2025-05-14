import cv2
import numpy as np


def sharpen_image(image: np.ndarray, strength: float = 1.5) -> np.ndarray:
    """
    Увеличивает резкость изображения с помощью фильтра повышения резкости.
    :param image: входное изображение (BGR)
    :param strength: сила резкости (1.0 — без изменений, >1 — резче)
    :return: улучшенное изображение
    """
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel * strength)


def reduce_noise(image: np.ndarray, h: int = 10) -> np.ndarray:
    """
    Уменьшает шумы на изображении с помощью алгоритма FastNlMeansDenoisingColored.
    :param image: входное изображение (BGR)
    :param h: параметр силы фильтрации
    :return: очищенное изображение
    """
    return cv2.fastNlMeansDenoisingColored(image, None, h=h, hColor=h, templateWindowSize=7, searchWindowSize=21)


def enhance_details(image: np.ndarray) -> np.ndarray:
    """
    Улучшает мелкие детали на изображении.
    :param image: входное изображение (BGR)
    :return: улучшенное изображение
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def equalize_exposure(image: np.ndarray) -> np.ndarray:
    """
    Выравнивает экспозицию изображения по каждому каналу.
    :param image: входное изображение (BGR)
    :return: выровненное изображение
    """
    channels = cv2.split(image)
    eq_channels = [cv2.equalizeHist(ch) for ch in channels]
    return cv2.merge(eq_channels)


def apply_clahe_filter(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """
    Применяет CLAHE к яркостному каналу.
    :param image: входное изображение (BGR, np.uint8)
    :return: улучшенное изображение (BGR, np.uint8)
    """
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Image must be a numpy array. Got {type(image)}")

    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected 3-channel image. Got shape: {image.shape}")

    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    enhanced_lab = cv2.merge((cl, a, b))
    result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    if not isinstance(result, np.ndarray):
        raise ValueError("CLAHE returned invalid type", result)

    if result.dtype != np.uint8:
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return result