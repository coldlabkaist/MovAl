from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QLineEdit, QSpinBox, QFileDialog, QTextEdit, QMessageBox
)
import yaml
import os

class YamlMaker(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("YAML Maker")
        self.setFixedSize(600, 700)

        self.data_dir = None 
        self.class_names = {}  
        self.kpt_names = []  

        main_layout = QVBoxLayout(self)

        self.load_data_btn = QPushButton("Load Splitted Data")
        self.load_data_btn.clicked.connect(self.load_splitted_data)
        main_layout.addWidget(self.load_data_btn)

        class_layout = QHBoxLayout()
        class_layout.addWidget(QLabel("Number of Classes:"))
        self.class_spin = QSpinBox()
        self.class_spin.setMinimum(1)
        self.class_spin.setMaximum(100)
        self.class_spin.valueChanged.connect(self.update_class_names)
        class_layout.addWidget(self.class_spin)
        main_layout.addLayout(class_layout)

        self.class_edit = QTextEdit()
        self.class_edit.setPlaceholderText("0: mouse1\n1: mouse2\n...")
        main_layout.addWidget(QLabel("Class Names (id: name):"))
        main_layout.addWidget(self.class_edit)

        main_layout.addWidget(QLabel("Keypoints Settings:"))

        self.nkpt_edit = QLineEdit()
        self.nkpt_edit.setPlaceholderText("e.g., 5")
        main_layout.addWidget(QLabel("nkpt:"))
        main_layout.addWidget(self.nkpt_edit)

        self.kpt_shape_edit = QLineEdit()
        self.kpt_shape_edit.setPlaceholderText("e.g., [5, 3]")
        main_layout.addWidget(QLabel("kpt_shape:"))
        main_layout.addWidget(self.kpt_shape_edit)

        self.flip_idx_edit = QLineEdit()
        self.flip_idx_edit.setPlaceholderText("e.g., [0,1,2,4,3]")
        main_layout.addWidget(QLabel("flip_idx:"))
        main_layout.addWidget(self.flip_idx_edit)

        self.kpt_names_edit = QTextEdit()
        self.kpt_names_edit.setPlaceholderText("e.g., Nose, Body_C, Tail, Ear_L, Ear_R")
        main_layout.addWidget(QLabel("kpt_names (comma separated):"))
        main_layout.addWidget(self.kpt_names_edit)

        self.make_yaml_btn = QPushButton("Make YAML")
        self.make_yaml_btn.clicked.connect(self.make_yaml)
        main_layout.addWidget(self.make_yaml_btn)

    def load_splitted_data(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Splitted Data Folder")
        if dir_path:
            self.folder_path = dir_path.replace("\\", "/")
            QMessageBox.information(self, "Loaded", f"Loaded data folder:\n{self.folder_path}")
            
    def update_class_names(self):
        num_classes = self.class_spin.value()
        text = ""
        for i in range(num_classes):
            text += f"{i}: mouse{i+1}\n"
        self.class_edit.setPlainText(text)

    def make_yaml(self):
        if not hasattr(self, 'folder_path'):
            QMessageBox.warning(self, "Error", "Please load splitted data folder first.")
            return

        try:
            class_lines = self.class_edit.toPlainText().splitlines()
            names_dict = {}
            for line in class_lines:
                if ':' in line:
                    idx, name = line.split(":", 1)
                    names_dict[int(idx.strip())] = name.strip()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Invalid class names input: {e}")
            return

        try:
            nkpt = int(self.nkpt_edit.text().strip())

            shape_values = [int(v.strip()) for v in self.kpt_shape_edit.text().replace("[", "").replace("]", "").split(",") if v.strip()]
            if len(shape_values) != 2:
                raise ValueError("kpt_shape must contain exactly two numbers.")
            kpt_shape = shape_values

            flip_idx = [int(v.strip()) for v in self.flip_idx_edit.text().replace("[", "").replace("]", "").replace(",", " ").split() if v.strip()]

            kpt_names = [name.strip() for name in self.kpt_names_edit.toPlainText().split(",") if name.strip()]
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Invalid Keypoints input: {e}")
            return

        data_yaml = {
            "train": f"{self.folder_path}/train/images",
            "val": f"{self.folder_path}/valid/images",
            "test": f"{self.folder_path}/test/images",
            "nc": len(names_dict),
            "names": names_dict,
            "nkpt": nkpt,
            "kpt_shape": kpt_shape,
            "flip_idx": flip_idx,
            "kpt_names": kpt_names,
        }

        save_path, _ = QFileDialog.getSaveFileName(self, "Save YAML", self.folder_path, "YAML Files (*.yaml *.yml)")
        if not save_path:
            return

        with open(save_path, "w") as f:
            lines = []
            lines.append(f"train: {data_yaml['train']}")
            lines.append(f"val: {data_yaml['val']}")
            lines.append(f"test: {data_yaml['test']}")
            lines.append(f"nc: {data_yaml['nc']}")
            lines.append("names:")
            for k, v in data_yaml['names'].items():
                lines.append(f"  {k}: {v}")
            lines.append(f"nkpt: {data_yaml['nkpt']}")
            lines.append(f"kpt_shape: {data_yaml['kpt_shape']}")
            lines.append(f"flip_idx: {data_yaml['flip_idx']}")
            lines.append(f"kpt_names: {data_yaml['kpt_names']}")
            
            f.write("\n".join(lines))