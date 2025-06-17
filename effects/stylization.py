from fastapi import UploadFile
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import torch.optim as optim
from effects.utils import smart_resize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===== Модель =====
class StyleTransferModel(nn.Module):
    def __init__(self):
        super().__init__()
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
                '21': 'conv4_2',
                '28': 'conv5_1'
            }
        features = {}
        x = image
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features


# ===== Трансформации =====
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ===== Функции =====
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    return torch.mm(tensor, tensor.t())


def im_convert(tensor):
    """
    Корректно денормализует и конвертирует тензор в изображение numpy
    """
    tensor = tensor.cpu().clone().detach().squeeze(0)  # [1,C,H,W] -> [C,H,W]
    
    # Обратная нормализация ДО преобразования в numpy
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    tensor = tensor * std + mean  # обратная нормализация

    tensor = tensor.clamp(0, 1)
    image = tensor.numpy().transpose(1, 2, 0)  # CHW -> HWC
    return (image * 255).astype(np.uint8)

def style_transfer(content_file: UploadFile, style_file: UploadFile, steps=1000, alpha=1, beta=1e7):
    try:
        # Загружаем изображения
        content_image = Image.open(content_file.file).convert("RGB")
        style_image = Image.open(style_file.file).convert("RGB")

        content_image = smart_resize(np.array(content_image), max_side=512)
        style_image = smart_resize(np.array(style_image), max_side=512)

        content_image = Image.fromarray(content_image)
        style_image = Image.fromarray(style_image)

        content_tensor = transform(content_image).unsqueeze(0).to(device)
        style_tensor = transform(style_image).unsqueeze(0).to(device)
        
        # Создаём модель
        transfer_model = StyleTransferModel().to(device)

        # Получаем фичи
        content_features = transfer_model.get_features(content_tensor)
        style_features = transfer_model.get_features(style_tensor)

        # Считаем грам матрицы
        style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

        target = content_tensor.clone().requires_grad_(True).to(device)

        optimizer = optim.Adam([target], lr=0.003)

        # Обучение
        for i in range(1, steps + 1):
            target_features = transfer_model.get_features(target)

            content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)

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

        # Конвертация обратно в numpy
        final_image = im_convert(target)
        final_image = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
        
        return final_image

    except Exception as e:
        raise RuntimeError(f"Style transfer failed: {str(e)}") from e
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

def load_image_from_upload(file: UploadFile) -> torch.Tensor:
    image = Image.open(file.file).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)