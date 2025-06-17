from effects.models import get_selfie_segmentation_model
import cv2
import numpy as np


def blur_background(image: np.ndarray, blur_intensity: int = 51) -> np.ndarray:
    selfie_segmentation = get_selfie_segmentation_model()

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = selfie_segmentation.process(rgb_image)
    mask = results.segmentation_mask > 0.5

    blurred = cv2.GaussianBlur(image, (blur_intensity, blur_intensity), 0)
    result = np.where(mask[..., None], image, blurred)
    return result


def remove_background(image: np.ndarray) -> np.ndarray:
    selfie_segmentation = get_selfie_segmentation_model()

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = selfie_segmentation.process(rgb_image)
    mask = results.segmentation_mask > 0.7

    alpha_channel = np.zeros(mask.shape, dtype=np.uint8)
    alpha_channel[mask] = 255
    b, g, r = cv2.split(image)
    return cv2.merge([b, g, r, alpha_channel])