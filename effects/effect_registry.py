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
    style_transfer as custom_style_transfer,
    apply_hdr_effect,
    pixelate,
    vignette_effect,
    film_grain_effect
)
from effects.background import (
    remove_background as custom_remove_background,
    blur_background as custom_blur_background
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
    equalize_exposure
)

# Регистрация всех доступных фильтров
EFFECT_REGISTRY = {
    # Basic
    "gaussian_blur": gaussian_blur,
    "median_blur": median_blur,
    "average_blur": average_blur,
    "bilateral_blur": bilateral_blur,
    "grayscale": grayscale,
    "invert_colors": invert_colors,
    "adjust_brightness": adjust_brightness,
    "adjust_contrast": adjust_contrast,

    # Artistic
    "pencil_sketch": pencil_sketch,
    "watercolor": apply_watercolor,
    "oil_painting": oil_painting_effect,
    "pencil_with_color": pencil_with_color_effect,
    "cartoonify": cartoonify,

    # Filters
    "warm_cold": apply_warm_cold_filter,
    "negative": apply_negative,
    "tint": apply_tint,
    "solarize": solarize,
    "black_white_negative": black_white_negative,

    # Stylization
    "hdr": apply_hdr_effect,
    "pixelate": pixelate,
    "vignette": vignette_effect,
    "film_grain": film_grain_effect,

    # Background
    "remove_background": custom_remove_background,
    "blur_background": custom_blur_background,

    # Edges
    "canny": canny_edge_detection,
    "sobel": sobel_edge_detection,
    "laplacian": laplacian_edge_detection,
    "scharr": scharr_edge_detection,
    "contours": simple_contour_detection,

    # Enhancements
    "sharpen": sharpen_image,
    "reduce_noise": reduce_noise,
    "enhance_details": enhance_details,
    "equalize_exposure": equalize_exposure
}