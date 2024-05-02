import cv2
import os
import numpy as np


class FaceDetector:
    def __init__(self, haarcascade_path):
        """
        Инициализация детектора лиц с использованием предобученных моделей Haar Cascade от OpenCV.
        Parameters:
            haarcascade_path (str): Путь к файлу XML с Haar Cascade моделью.
        """
        self.face_cascade = cv2.CascadeClassifier(haarcascade_path)

    def detect_faces(self, image_path):
        """
        Обнаруживает лица на изображении и возвращает координаты областей, содержащих лица.
        Parameters:
            image_path (str): Путь к изображению, на котором необходимо обнаружить лица.
        Returns:
            list: Список кортежей, каждый из которых содержит координаты (x, y, w, h) обнаруженного лица.
        """
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        return faces

    def extract_faces(self, image_path, save_path=None):
        """
        Извлекает и сохраняет лица, обнаруженные на изображении.
        Parameters:
            image_path (str): Путь к изображению для обработки.
            save_path (str): Папка для сохранения извлечённых лиц.
        Returns:
            list: Список извлеченных лиц в виде массивов numpy.
        """
        faces = self.detect_faces(image_path)
        image = cv2.imread(image_path)
        face_images = []
        for i, (x, y, w, h) in enumerate(faces):
            face = image[y : y + h, x : x + w]
            if save_path:
                face_filename = f"{save_path}/face_{i}.png"
                cv2.imwrite(face_filename, face)
            face_images.append(face)
        return face_images


if __name__ == "__main__":
    detector = FaceDetector(
        "haarcascade_frontalface_default.xml"
    )  # Укажите актуальный путь к файлу .xml
    detected_faces = detector.extract_faces(
        "path_to_your_image.jpg", save_path="path_to_save_faces"
    )
