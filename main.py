from pathlib import Path

from PyQt6.QtCore import QSize
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()

        # Define window title and size
        self.setWindowTitle("Mini Project 1")
        self.setFixedSize(QSize(900, 750))

        # Main layout
        self.main_layout = QVBoxLayout()
        self.main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Set the layout to the window
        widget = QWidget()
        widget.setLayout(self.main_layout)
        self.setCentralWidget(widget)


class Data:
    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = keras.datasets.mnist.load_data()
        print(f"x_train shape: {self.x_train.shape} - y_train shape: {self.y_train.shape}")


new_data = Data()
# Run the application
app = QApplication([])
app.setStyleSheet(Path('style.qss').read_text())
window = MainWindow()
window.show()
app.exec()
