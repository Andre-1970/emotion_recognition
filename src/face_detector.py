import cv2
import os


class FaceDetector:
    def __init__(self):
        """
        Инициализация детектора лиц с использованием предобученных моделей Haar Cascade от OpenCV.
        Автоматически настраивает пути для загрузки и сохранения данных.
        """
        self.base_path = "data/train"  # Путь к исходным данным
        self.save_dir = "data/processed"  # Путь для сохранения результатов
        cascade_path = "data/resources/haarcascade_frontalface_default.xml"  # Путь к файлу Haar Cascade
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def detect_faces(self, image):
        """
        Обнаруживает лица на изображении и возвращает координаты областей, содержащих лица.
        Parameters:
            image (numpy.ndarray): Исходное изображение в формате BGR.
        Returns:
            list of tuple: Список координат (x, y, w, h) для каждого обнаруженного лица.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        return faces

    def extract_faces(self, image_path, category):
        """
        Извлекает и сохраняет лица, обнаруженные на изображении.
        Parameters:
            image_path (str): Полный путь к исходному изображению.
            category (str): Категория изображения, используется для создания соответствующей поддиректории.
        Returns:
            list: Список изображений лиц, извлеченных из исходного изображения.
        """
        image = cv2.imread(image_path)
        faces = self.detect_faces(image)
        face_images = []
        for i, (x, y, w, h) in enumerate(faces):
            face = image[y : y + h, x : x + w]
            face_filename = os.path.join(self.save_dir, category, f"face_{i}.png")
            cv2.imwrite(face_filename, face)
            face_images.append(face)
        return face_images

    def process_images(self):
        """
        Обрабатывает все изображения в заданных категориях, сохраняя результаты в поддиректориях.
        """
        categories = [
            name
            for name in os.listdir(self.base_path)
            if os.path.isdir(os.path.join(self.base_path, name))
        ]
        for category in categories:
            image_dir = os.path.join(self.base_path, category)
            category_save_dir = os.path.join(self.save_dir, category)
            os.makedirs(
                category_save_dir, exist_ok=True
            )  # Создаем директорию для сохранения
            images = [
                img
                for img in os.listdir(image_dir)
                if img.lower().endswith(("png", "jpg", "jpeg"))
            ]
            for image_name in images:
                image_path = os.path.join(image_dir, image_name)
                self.extract_faces(image_path, category)
