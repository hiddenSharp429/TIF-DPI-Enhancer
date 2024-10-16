from PyQt6.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLabel, 
                             QWidget, QSpinBox, QGroupBox, QTextEdit, QSlider, QComboBox, QToolBar, QColorDialog)
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QMouseEvent, QColor, QAction
from processing.tif_reader import TIFReader
from processing.background_remover import BackgroundRemover
from processing.dpi_enhancer import DPIEnhancer
from utils.image_utils import adjust_gamma, sharpen_image
from processing.image_segmentation import ImageSegmenter
import cv2
import numpy as np
import os

class ImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.split_ratio = 0.5
        self.dragging = False
        self.show_comparison = False
        self.slider_width = 20
        self.slider_height = 40

    def mousePressEvent(self, event: QMouseEvent):
        if self.show_comparison:
            slider_x = int(self.width() * self.split_ratio) - self.slider_width // 2
            if slider_x <= event.position().x() <= slider_x + self.slider_width:
                self.dragging = True

    def mouseReleaseEvent(self, event: QMouseEvent):
        self.dragging = False

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.dragging and self.show_comparison:
            self.split_ratio = max(0.1, min(0.9, event.position().x() / self.width()))
            self.parent.update_split_image()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.pixmap() and self.show_comparison:
            painter = QPainter(self)
            pen = QPen(Qt.GlobalColor.white)
            pen.setWidth(2)
            painter.setPen(pen)
            x = int(self.width() * self.split_ratio)
            painter.drawLine(x, 0, x, self.height())
            
            # 绘制滑块
            slider_x = x - self.slider_width // 2
            slider_y = (self.height() - self.slider_height) // 2
            painter.fillRect(slider_x, slider_y, self.slider_width, self.slider_height, Qt.GlobalColor.white)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced TIF DPI Enhancer")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("""
            QMainWindow { background-color: #f0f0f0; }
            QLabel { color: #333333; }
            QPushButton { background-color: #4CAF50; color: white; padding: 5px; }
            QPushButton:hover { background-color: #45a049; }
            QGroupBox { border: 1px solid #cccccc; margin-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }
        """)

        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        # File selection
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout()
        self.file_label = QLabel("No file selected")
        self.select_file_button = QPushButton("Select TIF File")
        self.select_file_button.clicked.connect(self.select_file)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.select_file_button)
        file_group.setLayout(file_layout)
        left_layout.addWidget(file_group)

        # Image info
        info_group = QGroupBox("Image Information")
        info_layout = QVBoxLayout()
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        info_layout.addWidget(self.info_text)
        info_group.setLayout(info_layout)
        left_layout.addWidget(info_group)

        # DPI enhancement
        dpi_group = QGroupBox("DPI Enhancement")
        dpi_layout = QVBoxLayout()
        self.dpi_label = QLabel("Target DPI:")
        self.dpi_spinbox = QSpinBox()
        self.dpi_spinbox.setRange(72, 2400)
        self.dpi_spinbox.setValue(300)
        self.enhance_button = QPushButton("Enhance DPI")
        self.enhance_button.clicked.connect(self.enhance_dpi)
        dpi_layout.addWidget(self.dpi_label)
        dpi_layout.addWidget(self.dpi_spinbox)
        dpi_layout.addWidget(self.enhance_button)
        dpi_group.setLayout(dpi_layout)
        left_layout.addWidget(dpi_group)

        # Image processing
        processing_group = QGroupBox("Image Processing")
        processing_layout = QVBoxLayout()
        
        self.sharpness_label = QLabel("Sharpness:")
        self.sharpness_slider = QSlider(Qt.Orientation.Horizontal)
        self.sharpness_slider.setRange(0, 100)
        self.sharpness_slider.setValue(0)
        self.sharpness_slider.valueChanged.connect(self.update_preview)
        
        self.gamma_label = QLabel("Gamma:")
        self.gamma_slider = QSlider(Qt.Orientation.Horizontal)
        self.gamma_slider.setRange(1, 200)
        self.gamma_slider.setValue(100)
        self.gamma_slider.valueChanged.connect(self.update_preview)
        
        self.interpolation_label = QLabel("Interpolation Method:")
        self.interpolation_combo = QComboBox()
        self.interpolation_combo.addItems(["Nearest", "Bilinear", "Bicubic", "Lanczos"])
        self.interpolation_combo.currentTextChanged.connect(self.update_preview)
        
        processing_layout.addWidget(self.sharpness_label)
        processing_layout.addWidget(self.sharpness_slider)
        processing_layout.addWidget(self.gamma_label)
        processing_layout.addWidget(self.gamma_slider)
        processing_layout.addWidget(self.interpolation_label)
        processing_layout.addWidget(self.interpolation_combo)
        processing_group.setLayout(processing_layout)
        left_layout.addWidget(processing_group)

        # Save button
        self.save_button = QPushButton("Save Enhanced Image")
        self.save_button.clicked.connect(self.save_enhanced_image)
        self.save_button.setEnabled(False)
        left_layout.addWidget(self.save_button)

        # Status
        self.status_label = QLabel("")
        left_layout.addWidget(self.status_label)

        # 右侧容器
        right_container = QGroupBox("Image Display")
        right_container_layout = QVBoxLayout()

        # 添加工具栏
        toolbar = QToolBar()
        self.show_masks_action = QAction("Show Text/Image Masks", self)
        self.show_masks_action.setCheckable(True)
        self.show_masks_action.triggered.connect(self.toggle_masks)
        toolbar.addAction(self.show_masks_action)

        self.change_text_mask_color_action = QAction("Change Text Mask Color", self)
        self.change_text_mask_color_action.triggered.connect(self.change_text_mask_color)
        toolbar.addAction(self.change_text_mask_color_action)

        self.change_image_mask_color_action = QAction("Change Image Mask Color", self)
        self.change_image_mask_color_action.triggered.connect(self.change_image_mask_color)
        toolbar.addAction(self.change_image_mask_color_action)

        self.toggle_comparison_action = QAction("Toggle Comparison", self)
        self.toggle_comparison_action.setCheckable(True)
        self.toggle_comparison_action.triggered.connect(self.toggle_comparison)
        toolbar.addAction(self.toggle_comparison_action)

        toolbar.addAction("Future Feature")

        right_container_layout.addWidget(toolbar)

        # 图像显示
        self.image_label = ImageLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_container_layout.addWidget(self.image_label)

        right_container.setLayout(right_container_layout)
        right_layout.addWidget(right_container)

        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 2)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.file_path = None
        self.original_image = None
        self.processed_image = None
        self.text_mask = None
        self.image_mask = None
        self.text_mask_color = QColor(255, 0, 0, 64)  # 半透明红色
        self.image_mask_color = QColor(0, 255, 0, 64)  # 半透明绿色

        self.segmenter = ImageSegmenter()

    def select_file(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select TIF File", "", "TIF Files (*.tif *.tiff)")
        if file_path:
            self.file_path = file_path
            self.file_label.setText(f"Selected file: {file_path}")
            self.load_image()

    def load_image(self):
        try:
            tif_reader = TIFReader(self.file_path)
            self.original_dpi = tif_reader.get_dpi()
            self.original_image = tif_reader.read_image()
            self.processed_image = None

            self.update_info()
            self.display_image(self.original_image)
            self.save_button.setEnabled(False)
        except Exception as e:
            self.status_label.setText(f"Error loading image: {str(e)}")

    def update_info(self):
        info = f"File: {self.file_path}\n"
        info += f"Current DPI: {self.original_dpi}\n"
        info += f"Image Size: {self.original_image.shape[1]}x{self.original_image.shape[0]}\n"
        info += f"Color Channels: {self.original_image.shape[2]}"
        self.info_text.setText(info)

    def display_image(self, image):
        try:
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        except Exception as e:
            print(f"Error displaying image: {str(e)}")
            print(f"Image shape: {image.shape}")
            print(f"Image dtype: {image.dtype}")

    def enhance_dpi(self):
        if not self.file_path:
            self.status_label.setText("Please select a TIF file first.")
            return

        target_dpi = int(self.dpi_spinbox.value())
        original_dpi = int(self.original_dpi)
        interpolation = self.interpolation_combo.currentText()

        try:
            dpi_enhancer = DPIEnhancer()
            if self.text_mask is None or self.image_mask is None:
                self.segment_image()
            enhanced_image = dpi_enhancer.enhance(self.original_image, self.text_mask, self.image_mask, original_dpi, target_dpi, interpolation)

            self.processed_image = enhanced_image
            self.update_display()
            self.save_button.setEnabled(True)
            self.toggle_comparison_action.setEnabled(True)
            self.status_label.setText(f"DPI enhanced from {self.original_dpi} to {target_dpi}. Click 'Save Enhanced Image' to save.")
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")

    def update_preview(self):
        if not self.file_path or self.original_image is None:
            return

        target_dpi = int(self.dpi_spinbox.value())
        original_dpi = int(self.original_dpi)
        interpolation = self.interpolation_combo.currentText()

        try:
            dpi_enhancer = DPIEnhancer()
            if self.text_mask is None or self.image_mask is None:
                self.segment_image()
            enhanced_image = dpi_enhancer.enhance(self.original_image, self.text_mask, self.image_mask, original_dpi, target_dpi, interpolation)

            self.processed_image = enhanced_image
            self.update_display()
            self.save_button.setEnabled(True)
            self.toggle_comparison_action.setEnabled(True)
            self.status_label.setText(f"DPI enhanced from {self.original_dpi} to {target_dpi}. Click 'Save Enhanced Image' to save.")
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")

    def update_split_image(self):
        if self.original_image is None:
            return

        if self.processed_image is None or not self.image_label.show_comparison:
            display_image = self.original_image.copy()
        else:
            left_image = self.original_image
            right_image = self.processed_image

            if left_image.shape != right_image.shape:
                right_image = cv2.resize(right_image, (left_image.shape[1], left_image.shape[0]))

            height, width, channel = left_image.shape
            split_width = int(width * self.image_label.split_ratio)

            display_image = np.zeros((height, width, channel), dtype=np.uint8)
            display_image[:, :split_width] = left_image[:, :split_width]
            display_image[:, split_width:] = right_image[:, split_width:]

        if self.show_masks_action.isChecked() and self.text_mask is not None and self.image_mask is not None:
            mask_overlay = np.zeros_like(display_image, dtype=np.uint8)
            mask_overlay[self.text_mask > 0] = self.text_mask_color.getRgb()[:3]
            mask_overlay[self.image_mask > 0] = self.image_mask_color.getRgb()[:3]
            
            alpha = 0.5  # 调整这个值来改变遮罩的透明度
            display_image = cv2.addWeighted(display_image, 1, mask_overlay, alpha, 0)

        height, width, channel = display_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(display_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def save_enhanced_image(self):
        if self.processed_image is None:
            self.status_label.setText("No enhanced image to save.")
            return

        file_dialog = QFileDialog()
        output_path, _ = file_dialog.getSaveFileName(self, "Save Enhanced Image", "", "TIF Files (*.tif *.tiff)")
        if output_path:
            try:
                dpi_enhancer = DPIEnhancer()
                dpi_enhancer.save_image(self.processed_image, output_path, self.dpi_spinbox.value())
                self.status_label.setText(f"Enhanced image saved as {output_path}")
            except Exception as e:
                self.status_label.setText(f"Error saving image: {str(e)}")

    def toggle_masks(self):
        if self.original_image is not None:
            if self.show_masks_action.isChecked():
                self.segment_image()
            self.update_display()

    def segment_image(self):
        if self.original_image is not None:
            self.text_mask, self.image_mask = self.segmenter.segment_image(self.original_image)
            # 使用颜色信息进一步细化掩码
            self.text_mask, self.image_mask = self.segmenter.refine_masks(self.text_mask, self.image_mask, self.original_image)
            self.status_label.setText("Image segmentation completed.")

    def update_display(self):
        if self.original_image is None:
            return

        # 始终从原始图像开始
        display_image = self.original_image.copy()

        if self.show_masks_action.isChecked() and self.text_mask is not None and self.image_mask is not None:
            # 创建一个与原始图像相同大小的 RGBA 图像用于遮罩
            mask_overlay = np.zeros((*display_image.shape[:2], 4), dtype=np.uint8)
            
            # 为文字区域添加半透明遮罩
            text_color = self.text_mask_color.getRgb()
            mask_overlay[self.text_mask > 0] = (*text_color[:3], 64)  # 64 为四分之一透明度

            # 为图像区域添加半透明遮罩
            image_color = self.image_mask_color.getRgb()
            mask_overlay[self.image_mask > 0] = (*image_color[:3], 64)  # 64 为四分之一透明度

            # 将遮罩叠加到显示图像上
            alpha_mask = mask_overlay[:,:,3] / 255.0
            for c in range(3):  # 对每个颜色通道
                display_image[:,:,c] = display_image[:,:,c] * (1 - alpha_mask) + mask_overlay[:,:,c] * alpha_mask

        # 创建 QImage
        height, width, channel = display_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(display_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def change_text_mask_color(self):
        color = QColorDialog.getColor(self.text_mask_color, self, "Choose Text Mask Color", QColorDialog.ColorDialogOption.ShowAlphaChannel)
        if color.isValid():
            self.text_mask_color = color
            self.update_display()

    def change_image_mask_color(self):
        color = QColorDialog.getColor(self.image_mask_color, self, "Choose Image Mask Color", QColorDialog.ColorDialogOption.ShowAlphaChannel)
        if color.isValid():
            self.image_mask_color = color
            self.update_display()

    def toggle_comparison(self):
        if self.processed_image is not None:
            self.image_label.show_comparison = self.toggle_comparison_action.isChecked()
            self.update_split_image()
