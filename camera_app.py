import sys
import cv2
import os
import uuid
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QTabWidget, QListWidget, QListWidgetItem, QGridLayout, QScrollArea, QFrame
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap


# Ensure the required directories exist
class CameraApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Capture App")

        self.image_folder = "captured_images"
        self.faces_folder = "added_faces"
        os.makedirs(self.image_folder, exist_ok=True)
        os.makedirs(self.faces_folder, exist_ok=True)

        self.capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.image_label = QLabel("Camera Preview")
        self.image_label.setFixedSize(640, 480)
        self.image_label.setStyleSheet("background-color: #333; color: white;")
        self.image_label.setAlignment(Qt.AlignCenter)

        # Buttons
        self.start_btn = QPushButton("Start Camera")
        self.stop_btn = QPushButton("Stop Camera")
        self.capture_btn = QPushButton("Take Picture")
        self.add_person_btn = QPushButton("Add Person")

        self.start_btn.clicked.connect(self.start_camera)
        self.stop_btn.clicked.connect(self.stop_camera)
        self.capture_btn.clicked.connect(self.capture_image)
        self.add_person_btn.clicked.connect(self.add_person)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        button_layout.addWidget(self.capture_btn)
        button_layout.addWidget(self.add_person_btn)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.image_label)
        left_layout.addLayout(button_layout)

        # Tabs for images and faces
        self.tabs = QTabWidget()
        self.image_tab = self.create_scrollable_grid()
        self.faces_tab = self.create_scrollable_grid()
        self.tabs.addTab(self.image_tab["widget"], "Captured Images")
        self.tabs.addTab(self.faces_tab["widget"], "Added Persons")

        # Main layout
        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout)
        main_layout.addWidget(self.tabs)

        self.setLayout(main_layout)

        self.current_frame = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def create_scrollable_grid(self):
        widget = QWidget()
        layout = QGridLayout()
        scroll = QScrollArea()
        frame = QFrame()
        frame.setLayout(layout)
        scroll.setWidget(frame)
        scroll.setWidgetResizable(True)

        outer_layout = QVBoxLayout(widget)
        outer_layout.addWidget(scroll)

        return {"widget": widget, "layout": layout, "items": []}

    def start_camera(self):
        self.capture = cv2.VideoCapture(0)
        self.timer.start(30)

    def stop_camera(self):
        if self.capture:
            self.capture.release()
        self.timer.stop()
        self.image_label.clear()
        self.image_label.setText("Camera Preview")

    def update_frame(self):
        ret, frame = self.capture.read()
        if ret:
            self.current_frame = frame.copy()
            rgb_image = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1) 
            height, width, channel = rgb_image.shape
            bytes_per_line = 3 * width
            qt_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.image_label.setPixmap(pixmap)

    def capture_image(self):
        if self.current_frame is not None:
            filename = os.path.join(self.image_folder, f"{uuid.uuid4().hex}.jpg")
            cv2.imwrite(filename, self.current_frame)
            self.add_to_grid(self.image_tab, filename)

    def add_person(self):
        if self.current_frame is not None:
            gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                face = self.current_frame[y:y + h, x:x + w]
                face_filename = os.path.join(self.faces_folder, f"{uuid.uuid4().hex}.jpg")
                cv2.imwrite(face_filename, face)
                self.add_to_grid(self.faces_tab, face_filename)

    def add_to_grid(self, tab, image_path):
        row = len(tab["items"]) // 3
        col = len(tab["items"]) % 3
        label = QLabel()
        pixmap = QPixmap(image_path).scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pixmap)
        tab["layout"].addWidget(label, row, col)
        tab["items"].append(label)

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CameraApp()
    window.resize(1200, 600)
    window.show()
    sys.exit(app.exec_())
