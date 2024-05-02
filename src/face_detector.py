import cv2
import os


class FaceDetector:
    def __init__(self):
        """
        Инициализация детектора лиц с использованием предобученных моделей Haar Cascade от OpenCV.
        """
        # Указываем путь к XML-файлу с предобученной моделью Haar Cascade
        cascade_path = "../data/resources/haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def detect_faces(self, image):
        """
        Обнаруживает лица на изображении и возвращает координаты областей, содержащих лица.
        Parameters:
            image (numpy.ndarray): Исходное изображение в формате BGR.
        Returns:
            list of tuple: Список кортежей, где каждый кортеж содержит координаты (x, y, w, h) каждого обнаруженного лица.
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

    def extract_faces(self, image_path, save_path):
        """
        Извлекает и сохраняет лица, обнаруженные на изображении.
        Parameters:
            image_path (str): Полный путь к исходному изображению.
            save_path (str): Путь к директории, где будут сохранены извлеченные лица.
        Returns:
            list: Список изображений лиц, извлеченных из исходного изображения.
        """
        image = cv2.imread(image_path)
        faces = self.detect_faces(image)
        face_images = []
        for i, (x, y, w, h) in enumerate(faces):
            face = image[y : y + h, x : x + w]
            face_filename = os.path.join(save_path, f"face_{i}.png")
            cv2.imwrite(face_filename, face)
            face_images.append(face)
        return face_images


if __name__ == "__main__":
    detector = FaceDetector()
    base_path = "../data/train"
    categories = [
        name
        for name in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, name))
    ]
    for category in categories:
        image_dir = os.path.join(base_path, category)
        save_dir = f"../data/processed/{category}"
        os.makedirs(
            save_dir, exist_ok=True
        )  # Создание директории, если она не существует

        images = [
            img
            for img in os.listdir(image_dir)
            if img.lower().endswith(("png", "jpg", "jpeg"))
        ]
        for image_name in images:
            image_path = os.path.join(image_dir, image_name)
            detected_faces = detector.extract_faces(image_path, save_dir)
            print(
                f"Processed {len(detected_faces)} faces from {image_name} in category '{category}'."
            )
