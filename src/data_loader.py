import os
import numpy as np
from PIL import Image
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class DataLoader:
    def __init__(self, base_path, csv_file="train.csv"):
        """
        Инициализация DataLoader с базовым путем к данным и путем к CSV файлу с метками.

        Parameters:
            base_path (str): Путь к папке с изображениями.
            csv_file (str): Имя файла CSV, содержащего метки и пути к изображениям.
        """
        self.base_path = base_path
        self.labels_df = pd.read_csv(os.path.join(base_path, "..", csv_file))
        self.update_image_paths()

    def update_image_paths(self):
        """
        Обновление путей к изображениям в датафрейме, чтобы они были полными.
        """
        self.labels_df["image_path"] = self.labels_df["image_path"].apply(
            lambda x: os.path.join(self.base_path, x.split("/")[-2], x.split("/")[-1])
        )

    def load_image(self, path, size=(224, 224)):
        """
        Загрузка изображения, его ресайзинг и нормализация.

        Parameters:
            path (str): Путь к изображению.
            size (tuple): Желаемый размер изображения.

        Returns:
            numpy.ndarray: Нормализованное изображение.
        """
        image = Image.open(path)
        image = image.resize(size)
        image = np.array(image) / 255.0  # Преобразование пикселей к диапазону [0, 1]
        return image

    def load_dataset(self):
        """
        Загрузка всех изображений и преобразование меток в one-hot вектора.

        Returns:
            tuple: массив numpy с изображениями и массив numpy с one-hot encoded метками.
        """
        images = np.array(
            [self.load_image(path) for path in self.labels_df["image_path"]]
        )
        emotions = self.convert_to_onehot(self.labels_df["emotion"])
        return images, emotions

    def convert_to_onehot(self, emotions):
        """
        Преобразование меток эмоций в one-hot encoded вектора.

        Parameters:
            emotions (pandas.Series): Серия меток эмоций.

        Returns:
            numpy.ndarray: Массив one-hot encoded меток.
        """
        encoder = OneHotEncoder(sparse=False)
        emotions = np.array(emotions).reshape(-1, 1)
        onehot_encoded = encoder.fit_transform(emotions)
        return onehot_encoded