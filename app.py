import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Список классов для FashionMNIST
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Определяем модель
class AdvancedResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AdvancedResNet, self).__init__()
        from torchvision.models import resnet18
        self.resnet = resnet18(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = True  

        # Заменяем последний слой на собственный
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

# Загрузка модели
def load_model():
    model = AdvancedResNet(num_classes=10)  # Инициализируем модель
    model.load_state_dict(torch.load('best_advanced_resnet.pth', map_location=torch.device('cpu')))  # Загружаем веса
    model.eval()  # Устанавливаем модель в режим оценки
    return model

# Трансформация изображения
def process_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Изменяем размер изображения для ResNet
        transforms.Grayscale(num_output_channels=3),  # Преобразуем в трехканальное изображение
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Нормализация для трех каналов
    ])
    return transform(image).unsqueeze(0)


st.title("Clothes Classification")
st.write("Загрузите изображение, чтобы классифицировать его с помощью модели Advanced ResNet.")

uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Загруженное изображение", use_column_width=True)

    # Загрузка модели
    model = load_model()

    # Обработка изображения и предсказание
    input_tensor = process_image(image)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = output.max(1)

    # Выводим название класса
    predicted_class_name = class_names[predicted_class.item()]
    st.write(f"Предсказанный класс: {predicted_class_name}")
