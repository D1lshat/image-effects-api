import cv2
import numpy as np


def gaussian_blur(image: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def average_blur(image: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    return cv2.blur(image, (kernel_size, kernel_size))


def median_blur(image: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    return cv2.medianBlur(image, kernel_size)


def bilateral_blur(image: np.ndarray, diameter: int = 9,
                   sigma_color: int = 75, sigma_space: int = 75) -> np.ndarray:
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)


def grayscale(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def invert_colors(image: np.ndarray) -> np.ndarray:
    return cv2.bitwise_not(image)


def adjust_brightness(image: np.ndarray, brightness: int = 30) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype("float32")
    hsv[..., 2] = np.clip(hsv[..., 2] + brightness, 0, 255)
    return cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)


def adjust_contrast(image: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)