from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton,
    QTextEdit, QLabel,
    QDialog, QMessageBox, QSpinBox, QFileDialog, QGroupBox, QFormLayout, QSlider, QCheckBox, QLineEdit, QWidget
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import subprocess
import os
import json
import random
import shutil
import yaml
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from .yaml_maker import YamlMaker


class PrepareDataDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Prepare Data")
        self.setFixedSize(300, 150)

        main_layout = QVBoxLayout()

        step1_label = QLabel("Step 1")
        self.data_split_btn = QPushButton("Data Split")
        self.data_split_btn.setFixedHeight(40)
        self.data_split_btn.clicked.connect(self.open_data_split)
        main_layout.addWidget(step1_label)
        main_layout.addWidget(self.data_split_btn)

        step2_label = QLabel("Step 2")
        self.make_yaml_btn = QPushButton("Make YAML")
        self.make_yaml_btn.setFixedHeight(40)
        self.make_yaml_btn.clicked.connect(self.open_yaml_maker)
        main_layout.addWidget(step2_label)
        main_layout.addWidget(self.make_yaml_btn)

        self.setLayout(main_layout)
        
    def open_data_split(self):
        dialog= DataSplitDialog(self)
        dialog.exec()
        
    def open_yaml_maker(self):
        dialog = YamlMaker(self)
        dialog.exec()
   
class DataSplitDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Data Split")
        self.setFixedSize(500, 400)

        

        self.label_dir = ""
        self.image_dir = ""

        layout = QVBoxLayout(self)

        label_layout = QHBoxLayout()
        self.label_path_label = QLabel("No folder selected")
        self.load_label_btn = QPushButton("Load Labels")
        self.load_label_btn.setFixedWidth(100)
        self.load_label_btn.clicked.connect(self.load_labels)
        
        label_layout.addWidget(self.label_path_label)
        label_layout.addWidget(self.load_label_btn)
        layout.addLayout(label_layout)

        image_layout = QHBoxLayout()
        self.image_path_label = QLabel("No folder selected")
        self.load_image_btn = QPushButton("Load Images")
        self.load_image_btn.setFixedWidth(100)
        self.load_image_btn.clicked.connect(self.load_images)
        
        image_layout.addWidget(self.image_path_label)
        image_layout.addWidget(self.load_image_btn)
        layout.addLayout(image_layout)

        layout.addSpacing(40) 
        self.count_label = QLabel("Label number: 0 / Image number: 0")
        self.count_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        count_font = QFont()
        count_font.setPointSize(11) 
        self.count_label.setFont(count_font)
        layout.addWidget(self.count_label)

        layout.addSpacing(20) 

        ratio_layout = QFormLayout()

        self.train_slider = QSlider(Qt.Orientation.Horizontal)
        self.train_slider.setRange(0, 100)
        self.train_slider.setValue(70)
        self.train_spin = QSpinBox()
        self.train_spin.setRange(0, 100)
        self.train_spin.setValue(70)
        self.train_slider.valueChanged.connect(self.train_spin.setValue)
        self.train_spin.valueChanged.connect(self.train_slider.setValue)
        ratio_layout.addRow("Train %", self.create_slider_spinbox_layout(self.train_slider, self.train_spin))

        self.valid_slider = QSlider(Qt.Orientation.Horizontal)
        self.valid_slider.setRange(0, 100)
        self.valid_slider.setValue(20)
        self.valid_spin = QSpinBox()
        self.valid_spin.setRange(0, 100)
        self.valid_spin.setValue(20)
        self.valid_slider.valueChanged.connect(self.valid_spin.setValue)
        self.valid_spin.valueChanged.connect(self.valid_slider.setValue)
        ratio_layout.addRow("Valid %", self.create_slider_spinbox_layout(self.valid_slider, self.valid_spin))

        self.test_slider = QSlider(Qt.Orientation.Horizontal)
        self.test_slider.setRange(0, 100)
        self.test_slider.setValue(10)
        self.test_spin = QSpinBox()
        self.test_spin.setRange(0, 100)
        self.test_spin.setValue(10)
        self.test_slider.valueChanged.connect(self.test_spin.setValue)
        self.test_spin.valueChanged.connect(self.test_slider.setValue)
        ratio_layout.addRow("Test %", self.create_slider_spinbox_layout(self.test_slider, self.test_spin))

        layout.addLayout(ratio_layout)

        layout.addSpacing(20)

        self.run_btn = QPushButton("Run")
        layout.addWidget(self.run_btn)
        self.run_btn.clicked.connect(self.run_split)

        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

    def create_slider_spinbox_layout(self, slider, spinbox):
        hlayout = QHBoxLayout()
        hlayout.addWidget(slider)
        hlayout.addWidget(spinbox)
        return hlayout

    def load_labels(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Labels Folder")
        if not folder:
            return
        self.label_dir = folder
        self.label_path_label.setText(folder)
        self.update_count()

    def load_images(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Images Folder")
        if not folder:
            return
        self.image_dir = folder
        self.image_path_label.setText(folder)
        self.update_count()

    def update_count(self):
        label_count = len([f for f in os.listdir(self.label_dir) if f.endswith('.txt')]) if self.label_dir else 0
        image_count = len([f for f in os.listdir(self.image_dir) if f.endswith(('.jpg', '.png'))]) if self.image_dir else 0
        self.count_label.setText(f"Label number: {label_count} / Image number: {image_count}")
        
    def run_split(self):
        if not self.label_dir or not self.image_dir:
            QMessageBox.warning(self, "Error", "Please load both labels and images folder first.")
            return

        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if not output_dir:
            return

        image_files = [f for f in os.listdir(self.image_dir) if f.lower().endswith(('.jpg', '.png'))]
        label_files = [f for f in os.listdir(self.label_dir) if f.endswith('.txt')]

        image_basenames = set(os.path.splitext(f)[0] for f in image_files)
        label_basenames = set(os.path.splitext(f)[0] for f in label_files)

        valid_basenames = list(image_basenames & label_basenames)

        if not valid_basenames:
            QMessageBox.warning(self, "Error", "No matching image-label pairs found.")
            return

        random.shuffle(valid_basenames)

        train_ratio = self.train_spin.value() / 100
        valid_ratio = self.valid_spin.value() / 100

        total = len(valid_basenames)
        train_count = int(total * train_ratio)
        valid_count = int(total * valid_ratio)

        train_names = valid_basenames[:train_count]
        valid_names = valid_basenames[train_count:train_count+valid_count]
        test_names = valid_basenames[train_count+valid_count:]

        split_sets = {
            'train': train_names,
            'valid': valid_names,
            'test': test_names
        }

        for split, names in split_sets.items():
            img_dst = os.path.join(output_dir, split, 'images')
            lbl_dst = os.path.join(output_dir, split, 'labels')
            os.makedirs(img_dst, exist_ok=True)
            os.makedirs(lbl_dst, exist_ok=True)

            for name in names:
                for ext in ['.jpg', '.png']:
                    img_path = os.path.join(self.image_dir, name + ext)
                    if os.path.exists(img_path):
                        shutil.copy(img_path, os.path.join(img_dst, name + ext))
                        break  
                    
                label_path = os.path.join(self.label_dir, name + '.txt')
                shutil.copy(label_path, os.path.join(lbl_dst, name + '.txt'))

        QMessageBox.information(self, "Success",
                                f"✅ Data Split complete!\nTrain: {len(train_names)}\nValid: {len(valid_names)}\nTest: {len(test_names)}")
