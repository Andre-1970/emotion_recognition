import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore


class EmotionClassifier:
    def __init__(self):
        """
        Инициализация классификатора эмоций с загрузкой предобученной модели и настройкой генератора данных.
        """
        model_path = "models/mobilenet_v3.h5"
        self.model = load_model(model_path)
        self.image_size = (224, 224)  # Размер изображений

        # Настройка генератора данных с аугментацией для использования при предсказании
        self.datagen = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.mobilenet_v3.preprocess_input,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest",
        )

    def prepare_image(self, image_path):
        """
        Загружает и обрабатывает изображение для предсказания.
        Parameters:
            image_path (str): Путь к изображению для классификации.
        Returns:
            Обработанное изображение.
        """
        img = tf.keras.preprocessing.image.load_img(
            image_path, target_size=self.image_size
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Добавление размерности батча
        return img_array

    def predict_emotion(self, image_path):
        """
        Выполняет предсказание эмоции на изображении.
        Parameters:
            image_path (str): Путь к изображению.
        Returns:
            int: Индекс предсказанной эмоции.
        """
        image = self.prepare_image(image_path)
        image = self.datagen.standardize(image)  # Применение аугментации
        predictions = self.model.predict(image)
        predicted_index = np.argmax(predictions, axis=1)[0]
        return predicted_index
