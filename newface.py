from __future__ import annotations

import os
import sys
import time

import cv2
from PySide6.QtCore import Qt, QThread, Signal, Slot, QEvent
from PySide6.QtGui import QAction, QImage, QKeySequence, QPixmap
from PySide6.QtWidgets import (QApplication, QComboBox, QGroupBox, QHBoxLayout, QLabel, QMainWindow, QPushButton, QSizePolicy, QVBoxLayout, QWidget)


class Thread(QThread):
    updateFrame = Signal(QImage)
    updateCount = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.trained_file = None
        self.running = False
        self.cap = None

    def set_file(self, fname):
        base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))  # 실행 파일이 있는 디렉토리
        haarcascade_dir = os.path.join(base_dir, "haarcascades")  # 같은 디렉토리 내 haarcascades 폴더
        if not os.path.exists(haarcascade_dir):
            os.makedirs(haarcascade_dir)  # 폴더가 없으면 생성
            print(f"Warning: 'haarcascades' folder not found. Created an empty one at {haarcascade_dir}")
        
        self.trained_file = os.path.join(haarcascade_dir, fname)
        
        if not os.path.exists(self.trained_file):
            print(f"Error: Could not find XML file: {self.trained_file}")
            self.trained_file = None

    def run(self):
        if not self.trained_file:
            print("Error: No trained model file set.")
            return
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return
        
        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cascade = cv2.CascadeClassifier(self.trained_file)
            if cascade.empty():
                print(f"Error: Could not load cascade classifier from {self.trained_file}")
            else:
                detections = cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                self.updateCount.emit(len(detections))
                for (x, y, w, h) in detections:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = color_frame.shape
            img = QImage(color_frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
            scaled_img = img.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
            
            self.updateFrame.emit(scaled_img)
        
        self.cap.release()


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Patterns detection")
        self.setGeometry(100, 100, 800, 600)

        self.label = QLabel(self)
        self.label.setFixedSize(640, 480)
        self.count_label = QLabel("Detected objects: 0", self)
        
        self.th = Thread(self)
        self.th.updateFrame.connect(self.setImage)
        self.th.updateCount.connect(self.setCount)
        
        self.group_model = QGroupBox("Trained model")
        model_layout = QHBoxLayout()

        self.combobox = QComboBox()
        base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        haarcascade_dir = os.path.join(base_dir, "haarcascades")
        if not os.path.exists(haarcascade_dir):
            os.makedirs(haarcascade_dir)  # 폴더 생성
            print(f"Warning: 'haarcascades' folder not found. Created an empty one at {haarcascade_dir}")
        
        for xml_file in os.listdir(haarcascade_dir):
            if xml_file.endswith(".xml"):
                self.combobox.addItem(xml_file)
        
        model_layout.addWidget(QLabel("File:"))
        model_layout.addWidget(self.combobox)
        self.group_model.setLayout(model_layout)

        buttons_layout = QHBoxLayout()
        self.button1 = QPushButton("Start")
        self.button2 = QPushButton("Stop")
        self.screenshot_button = QPushButton("Capture") 
        
        buttons_layout.addWidget(self.button1)
        buttons_layout.addWidget(self.button2)
        buttons_layout.addWidget(self.screenshot_button) 
        
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.count_label)
        layout.addWidget(self.group_model)
        layout.addLayout(buttons_layout)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.button1.clicked.connect(self.start)
        self.button2.clicked.connect(self.kill_thread)
        self.screenshot_button.clicked.connect(self.saveScreenshot)
        self.combobox.currentTextChanged.connect(self.set_model)
        
        self.installEventFilter(self)

    @Slot()
    def set_model(self, text):
        self.th.set_file(text)

    @Slot()
    def kill_thread(self):
        if self.th.running:
            print("Stopping thread...")
            self.th.running = False
            self.th.wait()
            cv2.destroyAllWindows()
        
    @Slot()
    def start(self):
        if not self.th.running:
            print("Starting thread...")
            self.th.set_file(self.combobox.currentText())
            if self.th.trained_file:
                self.th.start()

    @Slot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    @Slot(int)
    def setCount(self, count):
        self.count_label.setText(f"Detected objects: {count}")

    def saveScreenshot(self):
        if self.label.pixmap():
            image = self.label.pixmap().toImage()
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            image.save(f"screenshot_{timestamp}.png")
            print(f"Screenshot saved: screenshot_{timestamp}.png")

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress and event.key() == Qt.Key_Space:
            self.saveScreenshot()
        return super().eventFilter(obj, event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec())