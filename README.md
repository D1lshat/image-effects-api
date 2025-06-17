# Image Effects API

API для обработки изображений с поддержкой множества эффектов:  
- Базовые фильтры  
- Художественные стили  
- Удаление фона  
- Стилизация (в том числе EDSR-PyTorch)  
- Обнаружение краёв  
- Поддержка GPU через PyTorch / TensorFlow

---

## 🧰 Используемые технологии

- [FastAPI](https://fastapi.tiangolo.com/ ) – высокопроизводительный асинхронный API
- [OpenCV](https://opencv.org/ ) – обработка изображений
- [MediaPipe](https://mediapipe.dev/ ) – удаление и размытие фона
- [PyTorch](https://pytorch.org/ ) – стилизация изображений
- [rembg](https://github.com/danielgatis/rembg ) – альтернатива MediaPipe
- [TensorFlow / Keras] – опционально, для некоторых эффектов

---

устанавливай pytorch 
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118 

ngrok start --config=ngrok.yml image-effects-api
