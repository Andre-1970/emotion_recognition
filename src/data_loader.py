import os
import numpy as np
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


class DataLoader:
    def __init__(self, base_path, csv_file="train.csv", test_size=0.2, random_state=42):
        """
        Инициализирует DataLoader, указывая путь к директории с изображениями и файлом с метками.
        Parameters:
            base_path (str): Путь к папке, содержащей изображения.
            csv_file (str): Имя файла CSV в указанной директории, содержащего метки и пути к изображениям.
            test_size (float): Доля данных, которая будет использоваться как валидационный набор.
            random_state (int): Число, используемое для гарантии воспроизводимости результатов разделения.
        """
        self.base_path = base_path
        self.labels_df = pd.read_csv(os.path.join(base_path, csv_file))
        self.update_image_paths()
        self.encoder = OneHotEncoder(sparse_output=False)
        self.prepare_encoder()

        self.train_df, self.val_df = train_test_split(
            self.labels_df, test_size=test_size, random_state=random_state
        )

    def update_image_paths(self):
        """
        Обновляет пути к изображениям в DataFrame, преобразуя относительные пути в абсолютные.
        """
        self.labels_df["image_path"] = self.labels_df["image_path"].apply(
            lambda x: os.path.abspath(
                os.path.join(self.base_path, "train", x.replace("./train/", ""))
            )
        )

    def load_image(self, path, size=(224, 224)):
        """
        Загружает изображение, изменяет его размер и нормализует значения пикселей.
        Parameters:
            path (str): Абсолютный путь к изображению.
            size (tuple): Желаемый размер изображения после изменения размера (ширина, высота).
        Returns:
            numpy.ndarray: Массив изображения с нормализованными значениями пикселей.
        """
        image = Image.open(path)
        image = image.resize(size)
        image = np.array(image) / 255.0
        return image

    def load_dataset(self, dataset_type="train", batch_size=None):
        """
        Загружает изображения и метки, преобразует метки в формат one-hot.
        Parameters:
            dataset_type (str): Тип набора данных, который должен быть загружен.
                Параметр может принимать значения "train" для обучающего набора или "val" для валидационного.
            batch_size (int, optional): Количество образцов, обрабатываемых за один проход во время обучения сети.
        Returns:
            Генератор, возвращающий кортежи (numpy.ndarray, numpy.ndarray) - массив изображений и массив one-hot encoded меток.
        """
        if dataset_type == "train":
            data_df = self.train_df
        else:
            data_df = self.val_df

        if batch_size:
            for start in range(0, len(data_df), batch_size):
                end = start + batch_size
                images_batch = [
                    self.load_image(x) for x in data_df["image_path"][start:end]
                ]
                emotions_batch = self.convert_to_onehot(data_df["emotion"][start:end])
                yield np.array(images_batch), emotions_batch
        else:
            images = [self.load_image(x) for x in data_df["image_path"]]
            emotions = self.convert_to_onehot(data_df["emotion"])
            return np.array(images), emotions

    def prepare_encoder(self):
        """Обучение OneHotEncoder на всех уникальных метках в датафрейме."""
        emotions = self.labels_df["emotion"].unique().reshape(-1, 1)
        self.encoder.fit(emotions)

    def convert_to_onehot(self, emotions):
        """
        Преобразует список меток эмоций в формат one-hot encoded.
        Parameters:
            emotions (pandas.Series): Список меток эмоций.
        Returns:
            numpy.ndarray: Массив one-hot encoded меток, где каждый ряд соответствует одному вектору меток.
        """
        emotions = np.array(emotions).reshape(-1, 1)
        onehot_encoded = self.encoder.transform(emotions)
        return onehot_encoded
