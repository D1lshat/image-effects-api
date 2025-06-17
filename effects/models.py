import mediapipe as mp
from functools import lru_cache
import logging
# Это тестовая функция которая должна генерировать изображение по описание. В проекте он не используется так как API для такой работы явялются платными
@lru_cache(maxsize=1)
def get_selfie_segmentation_model():
    """Возвращает закэшированную модель SelfieSegmentation"""
    logging.info("Loading MediaPipe SelfieSegmentation model...")
    return mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)


@lru_cache(maxsize=1)
def get_style_transfer_model():
    """Загружает и возвращает модель стилизации через VGG"""
    import torch
    from effects.stylization import StyleTransferModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = StyleTransferModel()
    return model.model.to(device)
