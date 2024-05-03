import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore


class EmotionClassifier:
    def __init__(self):
        """
        Инициализация классификатора эмоций с загрузкой предобученной модели и настройкой генератора данных.
        """
        # Загружаем модель, компиляция отключена для дальнейшего использования оптимизатора
        self.model = load_model("models/mobilenet_v3.keras", compile=False)

        # Компиляция модели с новым оптимизатором
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Размер изображений для модели
        self.image_size = (224, 224)

        # Определение классов, с которыми обучалась модель
        self.class_names = sorted(next(os.walk("data/train"))[1])

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

    def prepare_image(self, image_input):
        """
        Обрабатывает изображение для предсказания.
        Parameters:
            image_input (str или numpy.ndarray): Путь к изображению или само изображение для классификации.
        Returns:
            numpy.ndarray: Обработанное изображение с добавленным измерением батча.
        """
        if isinstance(image_input, str):
            # Загрузка изображения по пути
            image = tf.keras.preprocessing.image.load_img(
                image_input, target_size=self.image_size
            )
            image = tf.keras.preprocessing.image.img_to_array(image)
        elif isinstance(image_input, np.ndarray):
            # Обработка numpy-изображения
            image = tf.image.resize(image_input, self.image_size)
            image = np.array(image)
        else:
            raise ValueError(
                "Функция prepare_image ожидает путь к файлу или np.ndarray"
            )

        image = np.expand_dims(image, axis=0)  # Добавление измерения батча
        return image

    def predict_emotion(self, image_input):
        """
        Выполняет предсказание эмоции на изображении.
        Parameters:
            image_input (str или numpy.ndarray): Путь к изображению или само изображение.
        Returns:
            str: Название предсказанной эмоции.
        """
        image = self.prepare_image(image_input)
        image = self.datagen.standardize(image)  # Применение аугментации
        predictions = self.model.predict(image)
        predicted_index = np.argmax(predictions, axis=1)[0]
        return self.class_names[predicted_index]
