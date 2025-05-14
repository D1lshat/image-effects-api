import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2


class StyleTransferModel:
    def __init__(self):
        self.model = models.vgg19(pretrained=True).features
        for param in self.model.parameters():
            param.requires_grad_(False)

    def get_features(self, image, layers=None):
        if layers is None:
            layers = {
                '0': 'conv1_1',
                '5': 'conv2_1',
                '10': 'conv3_1',
                '19': 'conv4_1',
                '28': 'conv5_1'
            }
        features = {}
        x = image
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features


def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    return torch.mm(tensor, tensor.t())


def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image


def load_image(image_file, max_size=512, shape=None):
    image = Image.open(image_file.file).convert("RGB")

    if max(shape or image.size) > max_size:
        size = max_size
    else:
        size = shape or image.size

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = transform(image).unsqueeze(0)
    return image


def style_transfer(content_file, style_file, steps=1000, alpha=1, beta=1e6):
    content = load_image(content_file, max_size=256)
    style = load_image(style_file, shape=content.shape[-2:])

    model = StyleTransferModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.model = model.model.to(device)
    content = content.to(device)
    style = style.to(device)

    content_features = model.get_features(content)
    style_features = model.get_features(style)

    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    target = content.clone().requires_grad_(True).to(device)

    optimizer = torch.optim.Adam([target], lr=0.003)

    for i in range(1, steps + 1):
        target_features = model.get_features(target)

        content_loss = torch.mean((target_features['conv4_1'] - content_features['conv4_1']) ** 2)

        style_loss = 0
        for layer in style_grams:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            style_gram = style_grams[layer]
            layer_style_loss = torch.mean((target_gram - style_gram) ** 2)
            style_loss += layer_style_loss / (style_gram.shape[0] * style_gram.shape[1])

        total_loss = alpha * content_loss + beta * style_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Iteration {i}, Total Loss: {total_loss.item()}")

    final_image = im_convert(target)
    final_image = (final_image * 255).astype(np.uint8)
    final_image = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
    return final_image


def apply_hdr_effect(image: np.ndarray) -> np.ndarray:
    """Усиливает контрастность и детализацию."""
    return cv2.detailEnhance(image, sigma_s=12, sigma_r=0.15)


def pixelate(image: np.ndarray, block_size: int = 10) -> np.ndarray:
    """Эффект пикселизации (low-res)."""
    height, width = image.shape[:2]
    small = cv2.resize(image, (width // block_size, height // block_size), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)


def vignette_effect(image: np.ndarray, strength: float = 2.0) -> np.ndarray:
    if len(image.shape) < 3:
        raise ValueError("Input image must be at least 3 channels (BGR or RGB)")
    
    rows, cols = image.shape[:2]

    # Защита от слишком маленького размера
    sigma_x = max(1.0, cols / strength)
    sigma_y = max(1.0, rows / strength)

    kernel_x = cv2.getGaussianKernel(cols, int(sigma_x), cv2.CV_32F)
    kernel_y = cv2.getGaussianKernel(rows, int(sigma_y), cv2.CV_32F)
    kernel = kernel_y @ kernel_x.T  # внешнее произведение

    # Нормализация без деления на ноль
    norm_kernel = kernel / (np.max(kernel) + 1e-8)

    # Создаем маску и нормализуем её до [0..1]
    vignette_mask = cv2.normalize(norm_kernel, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    # Умножаем изображение на маску
    result = image.astype("float32") * vignette_mask[..., np.newaxis]
    return np.clip(result, 0, 255).astype(np.uint8)


def film_grain_effect(image: np.ndarray, intensity: float = 0.05) -> np.ndarray:
    """Имитирует зернистость плёночного фото."""
    noise = np.random.normal(0, intensity * 255, image.shape).astype(np.int32)
    return np.clip(image + noise, 0, 255).astype(np.uint8)