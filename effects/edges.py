import cv2
import numpy as np


def canny_edge_detection(image: np.ndarray, threshold1: int = 100, threshold2: int = 200) -> np.ndarray:
    """Обнаруживает края с помощью алгоритма Canny."""
    return cv2.Canny(image, threshold1=threshold1, threshold2=threshold2)


def sobel_edge_detection(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Обнаруживает края с помощью оператора Собеля."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=kernel_size)
    grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=kernel_size)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)


def laplacian_edge_detection(image: np.ndarray) -> np.ndarray:
    """Обнаруживает края с помощью Лапласиана."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return cv2.convertScaleAbs(laplacian)


def scharr_edge_detection(image: np.ndarray) -> np.ndarray:
    """Обнаруживает края с помощью оператора Шарра (более точный, чем Sobel)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    grad_x = cv2.convertScaleAbs(grad_x)
    grad_y = cv2.convertScaleAbs(grad_y)
    return cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)


def simple_contour_detection(image: np.ndarray) -> np.ndarray:
    """Выделяет основные контуры на изображении."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Создаем чёрное изображение для рисования контуров
    contour_image = np.zeros_like(gray)
    cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 1)
    return contour_image