import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
from PIL import Image
import os
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Загрузка собственного DataLoader
from data_loader import DataLoader

# Путь к сохраненной модели
model_path = "models/mobilenet_v3.keras"

# Путь к тестовым данным
test_data_path = "data/test_kaggle"

# Инициализация DataLoader для получения меток эмоций
data_path = "data"
data_loader = DataLoader(data_path)

# Загрузка модели
model = load_model(model_path)

# Загрузка списка тестовых изображений
test_images = [f for f in os.listdir(test_data_path) if f.endswith('.jpg')]


# Функция для загрузки и предобработки изображений
def load_and_preprocess_image(image_path, size=(224, 224)):
    image = Image.open(image_path)
    image = image.resize(size)
    image = np.array(image) / 255.0
    return image


# Предсказание эмоций для тестовых изображений
image_paths = []
emotions = []

for image_name in test_images:
    image_path = os.path.join(test_data_path, image_name)
    image = load_and_preprocess_image(image_path)
    image = np.expand_dims(image, axis=0)  # Добавление размерности для batch
    pred = model.predict(image)
    pred_emotion = np.argmax(pred, axis=1)
    image_paths.append(image_name)
    emotions.append(pred_emotion[0])

# Создание DataFrame для результатов
emotions_labels = data_loader.encoder.categories_[
    0]  # Метки эмоций из OneHotEncoder
results_df = pd.DataFrame({
    "image_path": image_paths,
    "emotion": [emotions_labels[i] for i in emotions]
})

# Сохранение результатов в CSV файл
results_df.to_csv("data/predicted_submission.csv", index=False)

# Загрузка данных из sample_submission.csv для сравнения
sample_submission_df = pd.read_csv("data/sample_submission.csv")

# Объединение данных для сравнения
merged_df = sample_submission_df.merge(results_df,
                                       on="image_path",
                                       suffixes=('_true', '_pred'))

# Оценка точности и вывод отчета классификации
print("Classification Report:")
print(
    classification_report(merged_df['emotion_true'],
                          merged_df['emotion_pred']))
print("Accuracy:",
      accuracy_score(merged_df['emotion_true'], merged_df['emotion_pred']))
print(
    "F1 Score (macro):",
    f1_score(merged_df['emotion_true'],
             merged_df['emotion_pred'],
             average="macro"))
