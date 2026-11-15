import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTableWidget,
    QTableWidgetItem, QMessageBox, QSlider, QComboBox, QSplitter,
    QStackedWidget, QGroupBox
)
from PyQt6.QtCore import Qt

# Proje modüllerimiz
from data_loader import read_fits_data, load_fits_from_file, get_wcs_coordinates, get_sample_fits_path
from preprocessing import compute_image_difference
from analysis import detect_candidate_changes, extract_patch_features
from ml_model import AstronomyAnomalyDetector


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=5, dpi=100):
        fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi, facecolor='#0e1117')
        self.ax.set_facecolor('#0e1117')
        super(MplCanvas, self).__init__(fig)


class PlanetXMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Planet X & SPHEREx - NASA Mission Control")
        self.setGeometry(50, 50, 1400, 900)

        self.header1 = None
        self.data1 = None
        self.data2 = None
        self.detector = AstronomyAnomalyDetector()

        self.init_ui()
        self.load_default_data()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- LEFT CONTROL PANEL ---
        left_layout = QVBoxLayout()

        group_files = QGroupBox("📁 Data Source")
        form_files = QVBoxLayout()
        self.btn_load1 = QPushButton("Load Epoch 1 FITS")
        self.btn_load1.clicked.connect(self.load_epoch1)
        form_files.addWidget(self.btn_load1)

        self.btn_load2 = QPushButton("Load Epoch 2 FITS")
        self.btn_load2.clicked.connect(self.load_epoch2)
        form_files.addWidget(self.btn_load2)
        group_files.setLayout(form_files)
        left_layout.addWidget(group_files)

        group_settings = QGroupBox("🎨 Visualization & Mode")
        form_settings = QVBoxLayout()

        form_settings.addWidget(QLabel("Color Palette (Colormap):"))
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems(["inferno", "viridis", "magma", "plasma", "gray"])
        self.combo_cmap.currentIndexChanged.connect(self.refresh_current_view)
        form_settings.addWidget(self.combo_cmap)

        form_settings.addWidget(QLabel("Select Analysis Mode:"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems([
            "1. Single Image & WCS",
            "2. Multi-Epoch Gallery",
            "3. Temporal Difference & Candidates",
            "4. Isolation Forest (Unsupervised)",
            "5. SPHEREx Deep Learning & AI Scan",
            "6. Photometric Light Curve"
        ])
        self.combo_mode.currentIndexChanged.connect(self.switch_mode)
        form_settings.addWidget(self.combo_mode)
        group_settings.setLayout(form_settings)
        left_layout.addWidget(group_settings)

        left_layout.addStretch()

        # --- RIGHT CONTENT AREA (STACKED WIDGET) ---
        self.stack = QStackedWidget()

        # Page 1: Single Image
        self.page_single = QWidget()
        self.init_page_single()
        self.stack.addWidget(self.page_single)

        # Page 2: Gallery
        self.page_gallery = QWidget()
        self.init_page_gallery()
        self.stack.addWidget(self.page_gallery)

        # Page 3: Temporal Difference
        self.page_diff = QWidget()
        self.init_page_diff()
        self.stack.addWidget(self.page_diff)

        # Page 4: Isolation Forest
        self.page_iforest = QWidget()
        self.init_page_iforest()
        self.stack.addWidget(self.page_iforest)

        # Page 5: AI & Batch Results
        self.page_ai = QWidget()
        self.init_page_ai()
        self.stack.addWidget(self.page_ai)

        # Page 6: Light Curve
        self.page_lc = QWidget()
        self.init_page_lc()
        self.stack.addWidget(self.page_lc)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        left_container = QWidget()
        left_container.setLayout(left_layout)
        left_container.setMaximumWidth(320)

        splitter.addWidget(left_container)
        splitter.addWidget(self.stack)
        main_layout.addWidget(splitter)

    def init_page_single(self):
        layout = QHBoxLayout(self.page_single)
        self.canvas_single = MplCanvas(self, width=6, height=6)
        layout.addWidget(self.canvas_single)

        self.lbl_meta = QLabel("Loading metadata...")
        self.lbl_meta.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.lbl_meta)

    def init_page_gallery(self):
        layout = QHBoxLayout(self.page_gallery)
        self.canvas_g1 = MplCanvas(self, width=5, height=5)
        self.canvas_g2 = MplCanvas(self, width=5, height=5)
        layout.addWidget(self.canvas_g1)
        layout.addWidget(self.canvas_g2)

    def init_page_diff(self):
        layout = QHBoxLayout(self.page_diff)
        self.canvas_diff = MplCanvas(self, width=6, height=6)
        layout.addWidget(self.canvas_diff)

        right_v = QVBoxLayout()
        right_v.addWidget(QLabel("Sigma Threshold:"))
        self.slider_sigma = QSlider(Qt.Orientation.Horizontal)
        self.slider_sigma.setRange(5, 50)
        self.slider_sigma.setValue(20)
        self.slider_sigma.valueChanged.connect(self.update_diff_view)
        right_v.addWidget(self.slider_sigma)

        self.table_diff = QTableWidget()
        self.table_diff.setColumnCount(3)
        self.table_diff.setHorizontalHeaderLabels(["X", "Y", "Diff Intensity"])
        right_v.addWidget(self.table_diff)

        self.btn_export_diff = QPushButton("📥 Export Candidates to CSV")
        self.btn_export_diff.setStyleSheet("background-color: #3b3b3b; color: white; padding: 6px;")
        self.btn_export_diff.clicked.connect(lambda: self.export_table_to_csv(self.table_diff, "candidates_report.csv"))
        right_v.addWidget(self.btn_export_diff)

        container = QWidget()
        container.setLayout(right_v)
        layout.addWidget(container)

    def init_page_iforest(self):
        layout = QHBoxLayout(self.page_iforest)
        self.canvas_if = MplCanvas(self, width=6, height=6)
        layout.addWidget(self.canvas_if)

        self.lbl_if_info = QLabel("Isolation Forest analysis pending...")
        layout.addWidget(self.lbl_if_info)

    def init_page_ai(self):
        layout = QVBoxLayout(self.page_ai)
        self.btn_run_ai = QPushButton("🚀 Run Live PyTorch Autoencoder Scan")
        self.btn_run_ai.setStyleSheet("background-color: #2b5b84; color: white; font-weight: bold; padding: 10px;")
        self.btn_run_ai.clicked.connect(self.run_live_ai_scan)
        layout.addWidget(self.btn_run_ai)

        self.table_ai = QTableWidget()
        self.table_ai.setColumnCount(3)
        self.table_ai.setHorizontalHeaderLabels(["X Pixel", "Y Pixel", "Anomaly Score (MSE)"])
        layout.addWidget(self.table_ai)

        self.btn_export_ai = QPushButton("📥 Export AI Anomalies to CSV")
        self.btn_export_ai.setStyleSheet("background-color: #3b3b3b; color: white; padding: 6px;")
        self.btn_export_ai.clicked.connect(lambda: self.export_table_to_csv(self.table_ai, "ai_anomalies_report.csv"))
        layout.addWidget(self.btn_export_ai)

    def init_page_lc(self):
        layout = QHBoxLayout(self.page_lc)
        self.canvas_lc = MplCanvas(self, width=6, height=6)
        layout.addWidget(self.canvas_lc)

        v_ctrl = QVBoxLayout()
        v_ctrl.addWidget(QLabel("X Coordinate:"))
        self.slider_x = QSlider(Qt.Orientation.Horizontal)
        self.slider_x.valueChanged.connect(self.update_light_curve)
        v_ctrl.addWidget(self.slider_x)

        v_ctrl.addWidget(QLabel("Y Coordinate:"))
        self.slider_y = QSlider(Qt.Orientation.Horizontal)
        self.slider_y.valueChanged.connect(self.update_light_curve)
        v_ctrl.addWidget(self.slider_y)

        c_w = QWidget()
        c_w.setLayout(v_ctrl)
        layout.addWidget(c_w)

    def plot_scaled_image(self, canvas, data, cmap, title):
        canvas.ax.clear()
        if data is not None:
            clean = data[~np.isnan(data)]
            vmin, vmax = np.percentile(clean, (1, 99)) if len(clean) > 0 else (None, None)
            canvas.ax.imshow(data, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
            canvas.ax.set_title(title, color='white')
        canvas.draw()

    def load_default_data(self):
        path = get_sample_fits_path()
        if path:
            self.header1, self.data1 = read_fits_data(path)
            if self.data1 is not None:
                self.data2 = self.data1.copy().astype(float)
                max_val = np.nanmax(self.data2) if not np.isnan(self.data2).all() else 1.0
                y_grid, x_grid = np.ogrid[:self.data2.shape[0], :self.data2.shape[1]]
                mask = (x_grid - self.data2.shape[1] // 4) ** 2 + (y_grid - self.data2.shape[0] // 4) ** 2 <= 20
                self.data2[mask] += max_val * 0.8

                self.slider_x.setRange(0, self.data1.shape[1] - 1)
                self.slider_x.setValue(self.data1.shape[1] // 2)
                self.slider_y.setRange(0, self.data1.shape[0] - 1)
                self.slider_y.setValue(self.data1.shape[0] // 2)

                self.refresh_current_view()

    def load_epoch1(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Epoch 1 FITS", "", "FITS (*.fits *.fit)")
        if file_name:
            with open(file_name, "rb") as f:
                class DF:
                    def __init__(self, c): self.c = c

                    def read(self): return self.c

                self.header1, self.data1 = load_fits_from_file(DF(f.read()))
            self.refresh_current_view()

    def load_epoch2(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Epoch 2 FITS", "", "FITS (*.fits *.fit)")
        if file_name:
            with open(file_name, "rb") as f:
                class DF:
                    def __init__(self, c): self.c = c

                    def read(self): return self.c

                _, self.data2 = load_fits_from_file(DF(f.read()))
            self.refresh_current_view()

    def switch_mode(self, index):
        self.stack.setCurrentIndex(index)
        self.refresh_current_view()

    def refresh_current_view(self):
        if self.data1 is None or self.data2 is None:
            return

        idx = self.stack.currentIndex()
        cmap = self.combo_cmap.currentText()

        if idx == 0:  # Single Image
            self.plot_scaled_image(self.canvas_single, self.data1, cmap, "Epoch 1 Image")
            meta_txt = f"Shape: {self.data1.shape}\nMin: {np.nanmin(self.data1):.2f}\nMax: {np.nanmax(self.data1):.2f}"
            self.lbl_meta.setText(meta_txt)

        elif idx == 1:  # Gallery
            self.plot_scaled_image(self.canvas_g1, self.data1, cmap, "Epoch 1")
            self.plot_scaled_image(self.canvas_g2, self.data2, cmap, "Epoch 2 (Anomalous)")

        elif idx == 2:  # Temporal Difference
            self.update_diff_view()

        elif idx == 3:  # Isolation Forest
            self.canvas_if.ax.clear()
            diff = compute_image_difference(self.data1, self.data2)
            features, coords = extract_patch_features(diff, window_size=7)
            if len(features) > 0:
                preds, scores = self.detector.fit_predict(features)
                outliers = np.where(preds == -1)[0]
                self.canvas_if.ax.imshow(diff, cmap="bwr", origin="lower")
                if len(outliers) > 0:
                    pts = [coords[i] for i in outliers]
                    self.canvas_if.ax.scatter([p[0] for p in pts], [p[1] for p in pts], color='lime', s=20)
                self.canvas_if.ax.set_title(f"Isolation Forest Anomalies ({len(outliers)})", color='white')
                self.canvas_if.draw()

        elif idx == 5:  # Light Curve
            self.update_light_curve()

    def update_diff_view(self):
        if self.data1 is None or self.data2 is None: return
        sigma = self.slider_sigma.value() / 10.0
        diff = compute_image_difference(self.data1, self.data2)
        candidates, threshold = detect_candidate_changes(diff, threshold_sigma=sigma)

        self.canvas_diff.ax.clear()
        clean_diff = diff[~np.isnan(diff)]
        if len(clean_diff) > 0:
            vmax = np.percentile(np.abs(clean_diff), 98)
            if vmax == 0: vmax = np.nanmax(np.abs(clean_diff))
            if vmax == 0: vmax = 1.0
        else:
            vmax = 1.0

        self.canvas_diff.ax.imshow(diff, cmap="bwr", origin="lower", vmin=-vmax, vmax=vmax)

        if len(candidates) > 0:
            self.canvas_diff.ax.scatter([c["x"] for c in candidates], [c["y"] for c in candidates], color='yellow',
                                        s=15, marker='x')

        self.canvas_diff.ax.set_title(f"Difference Map (Sigma: {sigma})", color='white')
        self.canvas_diff.draw()

        self.table_diff.setRowCount(len(candidates))
        for r, c in enumerate(candidates[:100]):
            self.table_diff.setItem(r, 0, QTableWidgetItem(str(c["x"])))
            self.table_diff.setItem(r, 1, QTableWidgetItem(str(c["y"])))
            self.table_diff.setItem(r, 2, QTableWidgetItem(f"{c['intensity_diff']:.2f}"))

    def run_live_ai_scan(self):
        if self.data2 is None: return
        patches, coords = self.detector.extract_patches(self.data2, patch_size=32)
        if len(patches) > 0:
            if not self.detector.is_trained:
                self.detector.train_autoencoder(patches, epochs=3)
            anomalies, _ = self.detector.detect_anomalies_ae(patches, coords)

            self.table_ai.setRowCount(len(anomalies))
            for r, a in enumerate(anomalies):
                self.table_ai.setItem(r, 0, QTableWidgetItem(str(a["x"])))
                self.table_ai.setItem(r, 1, QTableWidgetItem(str(a["y"])))
                self.table_ai.setItem(r, 2, QTableWidgetItem(f"{a['score']:.5f}"))
            QMessageBox.information(self, "AI Scan Success", f"{len(anomalies)} anomalies detected successfully.")

    def update_light_curve(self):
        if self.data1 is None or self.data2 is None: return
        x = self.slider_x.value()
        y = self.slider_y.value()

        self.canvas_lc.ax.clear()
        epochs = ["Epoch 1", "Epoch 2"]
        fluxes = [float(self.data1[y, x]), float(self.data2[y, x])]

        self.canvas_lc.ax.plot(epochs, fluxes, marker='o', color='cyan', linewidth=2)
        self.canvas_lc.ax.set_title(f"Light Curve ({x}, {y})", color='white')
        self.canvas_lc.ax.set_ylabel("Flux", color='white')
        self.canvas_lc.draw()

    def export_table_to_csv(self, table_widget, default_filename):
        if table_widget.rowCount() == 0:
            QMessageBox.warning(self, "Export Warning", "No data available in the table to export!")
            return

        file_name, _ = QFileDialog.getSaveFileName(self, "Save CSV Report", default_filename, "CSV Files (*.csv)")
        if file_name:
            try:
                with open(file_name, "w", encoding="utf-8") as f:
                    headers = [table_widget.horizontalHeaderItem(i).text() for i in range(table_widget.columnCount())]
                    f.write(",".join(headers) + "\n")

                    for row in range(table_widget.rowCount()):
                        row_data = []
                        for col in range(table_widget.columnCount()):
                            item = table_widget.item(row, col)
                            row_data.append(item.text() if item else "")
                        f.write(",".join(row_data) + "\n")

                QMessageBox.information(self, "Export Successful", f"Report successfully saved to:\n{file_name}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"An error occurred while saving the file:\n{e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PlanetXMainWindow()
    window.show()
    sys.exit(app.exec())