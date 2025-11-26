import sys
import time
import queue
import datetime
import cv2
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QSpinBox, QDoubleSpinBox, 
                             QPushButton, QGroupBox, QComboBox, QMessageBox,
                             QCheckBox)
import pyPOACamera

# Enable antialiasing for prettier plots
# If performance is low we can remove the antialias
#pg.setConfigOptions(antialias=True, imageAxisOrder='row-major')

class CameraWorker(QThread):
    image_ready = pyqtSignal(object)
    status_message = pyqtSignal(str)
    error_message = pyqtSignal(str)
    fps_updated = pyqtSignal(float)
    histogram_ready = pyqtSignal(object)
    
    # Profiling signals
    profile_data = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.running = False
        self.camera_id = 0
        self.camera_opened = False
        self.exposure_us = 10000 # 10ms default
        self.gain = 220 # Just above the low readout noise threshold
        self.auto_gain = False
        self.frame_limit = 0 # 0 means no limit
        self.mock_mode = False
        self.mock_width = 3856
        self.mock_height = 2180
        self.mock_cache = None
        self.requested_format = None
        self.current_format = pyPOACamera.POAImgFormat.POA_RAW8
        self.binning = 2 # Default binning 2
        
        # FPS Calculation
        self.last_fps_time = time.time()
        self.fps_frame_count = 0
        
        self.compute_histogram = False

    def run(self):
        self.running = True
        
        # Attempt to connect to camera
        count = pyPOACamera.GetCameraCount()
        if count > 0:
            err, props = pyPOACamera.GetCameraProperties(0)
            if err == pyPOACamera.POAErrors.POA_OK:
                self.camera_id = props.cameraID
                err = pyPOACamera.OpenCamera(self.camera_id)
                if err == pyPOACamera.POAErrors.POA_OK:
                    pyPOACamera.InitCamera(self.camera_id)
                    self.camera_opened = True
                    self.status_message.emit(f"Connected to {props.cameraModelName.decode('utf-8')}")
                    
                    # Initial config
                    # If a format was requested before start, use it. Otherwise use default.
                    if self.requested_format is not None:
                        self.current_format = self.requested_format
                    
                    pyPOACamera.SetImageFormat(self.camera_id, self.current_format)
                    pyPOACamera.SetImageBin(self.camera_id, self.binning)
                    # Use max resolution
                    pyPOACamera.SetImageSize(self.camera_id, props.maxWidth, props.maxHeight)
                    self.status_message.emit(f"Resolution: {props.maxWidth}x{props.maxHeight}")
                    # Set auto white balance
                    pyPOACamera.SetConfig(self.camera_id, pyPOACamera.POAConfig.POA_WB_R, 0, True)
                else:
                    self.error_message.emit(f"Failed to open camera: {pyPOACamera.GetErrorString(err)}")
                    self.mock_mode = True
            else:
                 self.error_message.emit("Failed to get camera properties.")
                 self.mock_mode = True
        else:
            self.status_message.emit("No camera found. Entering Mock Mode.")
            self.mock_mode = True

        if self.camera_opened:
            # Start continuous exposure
            pyPOACamera.StartExposure(self.camera_id, False) # False = Video Mode

        while self.running:
            if self.camera_opened:
                self._handle_camera_capture()
            elif self.mock_mode:
                if self.requested_format is not None:
                    self.current_format = self.requested_format
                    self.requested_format = None
                    self.status_message.emit(f"Mock Format changed to {self.current_format.name}")
                self._handle_mock_capture()
            else:
                time.sleep(0.1)

        # Cleanup
        if self.camera_opened:
            pyPOACamera.StopExposure(self.camera_id)
            pyPOACamera.CloseCamera(self.camera_id)
            self.camera_opened = False
            self.status_message.emit("Camera closed.")

    def _handle_camera_capture(self):
        # Apply current settings
        # Note: In a real high-perf app, we might only set these when changed.
        # For simplicity, we set them if they differ from camera state or just set them.
        # Checking current state first is better to avoid I/O overhead.
        
        # Exposure
        err, curr_exp, auto = pyPOACamera.GetExp(self.camera_id)
        if curr_exp != self.exposure_us:
            pyPOACamera.SetExp(self.camera_id, int(self.exposure_us), False)
            
        # Gain
        err, curr_gain, auto = pyPOACamera.GetGain(self.camera_id)
        # If auto state changed or (manual mode and value changed)
        if auto != self.auto_gain or (not self.auto_gain and curr_gain != self.gain):
            pyPOACamera.SetGain(self.camera_id, int(self.gain), self.auto_gain)

        # Frame Limit
        # Note: POA_FRAME_LIMIT might not support GetConfig in the same way or we just set it.
        # Let's try to get it first to avoid redundant sets.
        err, curr_limit, _ = pyPOACamera.GetConfig(self.camera_id, pyPOACamera.POAConfig.POA_FRAME_LIMIT)
        if curr_limit != self.frame_limit:
             pyPOACamera.SetConfig(self.camera_id, pyPOACamera.POAConfig.POA_FRAME_LIMIT, int(self.frame_limit), False)

        # Binning Change
        # Check current binning
        err, curr_bin = pyPOACamera.GetImageBin(self.camera_id)
        if curr_bin != self.binning:
            pyPOACamera.StopExposure(self.camera_id)
            err = pyPOACamera.SetImageBin(self.camera_id, int(self.binning))
            if err == pyPOACamera.POAErrors.POA_OK:
                self.status_message.emit(f"Binning changed to {self.binning}")
            else:
                self.error_message.emit(f"Failed to set binning: {pyPOACamera.GetErrorString(err)}")
            pyPOACamera.StartExposure(self.camera_id, False)

        # Format Change
        if self.requested_format is not None and self.requested_format != self.current_format:
            pyPOACamera.StopExposure(self.camera_id)
            err = pyPOACamera.SetImageFormat(self.camera_id, self.requested_format)
            if err == pyPOACamera.POAErrors.POA_OK:
                self.current_format = self.requested_format
                self.status_message.emit(f"Format changed to {self.current_format.name}")
            else:
                self.error_message.emit(f"Failed to set format: {pyPOACamera.GetErrorString(err)}")
            pyPOACamera.StartExposure(self.camera_id, False)
            self.requested_format = None

        # Poll for image
        err, is_ready = pyPOACamera.ImageReady(self.camera_id)
        if is_ready:
            # Get dimensions
            err, w, h = pyPOACamera.GetImageSize(self.camera_id)
            err, fmt = pyPOACamera.GetImageFormat(self.camera_id)
            size = pyPOACamera.ImageCalcSize(h, w, fmt)
            
            # Create buffer
            buf = np.zeros(size, dtype=np.uint8)
            
            # Fetch data
            err = pyPOACamera.GetImageData(self.camera_id, buf, 1000)
            if err == pyPOACamera.POAErrors.POA_OK:
                img = pyPOACamera.ImageDataConvert(buf, h, w, fmt)
                # Ensure we have a valid image for pyqtgraph (transpose might be needed depending on sensor)
                # pyqtgraph by default expects [x, y] or [x, y, rgb], but we changed the settings to row-major
                # such that it now expects [y, x, rgb]
                # This way we don't need to transpose the image
                if img is not None:
                    # Squeeze to remove single channel dim if mono
                    if img.shape[2] == 1:
                        img = img.squeeze(2)
                        self.image_ready.emit(img)
                    else:
                        self.image_ready.emit(img)
                    
                    if self.compute_histogram:
                        self._calculate_and_emit_histogram(img)
                        
                    self._measure_fps()
            else:
                # Timeout or error
                pass
        else:
            time.sleep(0.005) # Small sleep to prevent busy loop burning CPU

    def _handle_mock_capture(self):
        # Adjust mock dimensions based on binning
        # Assuming mock_width/height are full resolution
        w = self.mock_width // self.binning
        h = self.mock_height // self.binning
        
        t = time.time()
        x = int((np.sin(t) + 1) / 2 * (w - 100 // self.binning))
        y = int((np.cos(t) + 1) / 2 * (h - 100 // self.binning))
        

        if self.current_format == pyPOACamera.POAImgFormat.POA_RAW16:
            # 16-bit noise - optimize with randint
            noise = np.random.randint(25000, 35000, size=(h, w), dtype=np.uint16)
            noise[y:y+50//self.binning, x:x+50//self.binning] = 65535
        elif self.current_format == pyPOACamera.POAImgFormat.POA_RGB24:
            # RGB noise (Width, Height, 3) for pyqtgraph - optimize with randint
            #noise = np.random.randint(80, 120, size=(h, w, 3), dtype=np.uint8)
            noise = np.ones((h, w, 3), dtype=np.uint8)*80
            # Give different means to each channel (simulate by adding offset, but clip)
            # Faster to just generate with different ranges if we want distinct colors, 
            # but simple noise is fine. Let's just add offsets safely or ignore for speed.
            # To be safe and fast:
            # noise[:, :, 0] += 0
            # noise[:, :, 1] += 50
            # noise[:, :, 2] += 100
            # The above is slow due to broadcasting and clipping.
            # Let's just color the circle.
            
            # Colored circle (e.g., Red)
            noise[y:y+50//self.binning, x:x+50//self.binning, 0] = 255 # R
            noise[y:y+50//self.binning, x:x+50//self.binning, 1] = 0   # G
            noise[y:y+50//self.binning, x:x+50//self.binning, 2] = 0   # B
        else:
            # 8-bit mono (RAW8, MONO8)
            noise = np.random.randint(80, 120, size=(h, w), dtype=np.uint8)
            noise[y:y+50//self.binning, x:x+50//self.binning] = 255
    
        self.mock_cache = noise

        self.image_ready.emit(noise)
        
        if self.compute_histogram:
            self._calculate_and_emit_histogram(noise)
        
        self._measure_fps()
            
        time.sleep(max(0.01, self.exposure_us / 1000000.0)) # Simulate exposure time delay

    def _measure_fps(self):
        self.fps_frame_count += 1
        now = time.time()
        dt = now - self.last_fps_time
        if dt >= 1.0:
            fps = self.fps_frame_count / dt
            self.fps_updated.emit(fps)
            self.fps_frame_count = 0
            self.last_fps_time = now

    def set_histogram_enabled(self, enabled):
        self.compute_histogram = enabled

    def _calculate_and_emit_histogram(self, img):
        # Downsample for speed (calculate on 1% of pixels)
        step = 10
        
        # Check if color or mono
        is_color = len(img.shape) == 3 and img.shape[2] == 3
        
        data = {}
        
        if is_color:
            data['type'] = 'color'
            
            # R
            r_data = img[::step,::step,0]
            y_r, x_r = np.histogram(r_data, bins=256, range=(0, 255))
            data['r'] = (x_r[:-1], y_r)
            r_mean = np.mean(r_data)
            r_std = np.std(r_data)
            
            # G
            g_data = img[::step,::step,1]
            y_g, x_g = np.histogram(g_data, bins=256, range=(0, 255))
            data['g'] = (x_g[:-1], y_g)
            g_mean = np.mean(g_data)
            g_std = np.std(g_data)
            
            # B
            b_data = img[::step,::step,2]
            y_b, x_b = np.histogram(b_data, bins=256, range=(0, 255))
            data['b'] = (x_b[:-1], y_b)
            b_mean = np.mean(b_data)
            b_std = np.std(b_data)
            
            data['stats'] = (f"R: Mean={r_mean:.2f}, Std={r_std:.2f} | "
                             f"G: Mean={g_mean:.2f}, Std={g_std:.2f} | "
                             f"B: Mean={b_mean:.2f}, Std={b_std:.2f}")
            
        else:
            data['type'] = 'mono'
            
            # Mono
            # Determine range based on dtype
            if img.dtype == np.uint16:
                bins = 256 
                rng = (0, 65535)
            else:
                bins = 256
                rng = (0, 255)
            
            # Downsample
            sub_img = img[::step,::step]
            
            y, x = np.histogram(sub_img, bins=bins, range=rng)
            data['mono'] = (x[:-1], y)
            
            mean = np.mean(sub_img)
            std = np.std(sub_img)
            data['stats'] = f"Mono: Mean={mean:.2f}, Std={std:.2f}"
            
        self.histogram_ready.emit(data)

    def stop(self):
        self.running = False
        self.wait()

    def set_exposure(self, us):
        self.exposure_us = us

    def set_gain(self, val, auto=False):
        self.gain = val
        self.auto_gain = auto

    def set_frame_limit(self, val):
        self.frame_limit = val

    def set_binning(self, val):
        self.binning = val

    def set_format(self, fmt_name):
        try:
            # Convert string name back to enum
            # Assuming fmt_name is like "POA_RAW8"
            fmt = pyPOACamera.POAImgFormat[fmt_name]
            self.requested_format = fmt
            self.mock_cache = None
        except KeyError:
            pass


class HistogramWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Histogram")
        self.resize(600, 500)
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Statistics Label
        self.stats_label = QLabel("Statistics: N/A")
        self.stats_label.setStyleSheet("font-family: monospace;")
        layout.addWidget(self.stats_label)
        
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)
        
        self.plot_widget.setLabel('left', 'Count')
        self.plot_widget.setLabel('bottom', 'Pixel Value')
        self.plot_widget.showGrid(x=True, y=True)
        
        # Curves for RGB and Mono
        # Use fillLevel=0 and brush to fill area under curve
        self.curve_mono = self.plot_widget.plot(pen='w', fillLevel=0, brush=(255, 255, 255, 50))
        self.curve_r = self.plot_widget.plot(pen='r', fillLevel=0, brush=(255, 0, 0, 50))
        self.curve_g = self.plot_widget.plot(pen='g', fillLevel=0, brush=(0, 255, 0, 50))
        self.curve_b = self.plot_widget.plot(pen='b', fillLevel=0, brush=(0, 0, 255, 50))
        
        self.curves = [self.curve_mono, self.curve_r, self.curve_g, self.curve_b]
        
    def update_data(self, data):
        if not self.isVisible():
            return

        if data['type'] == 'color':
            self.curve_mono.clear()
            
            self.curve_r.setData(data['r'][0], data['r'][1])
            self.curve_g.setData(data['g'][0], data['g'][1])
            self.curve_b.setData(data['b'][0], data['b'][1])
            
        else:
            self.curve_r.clear()
            self.curve_g.clear()
            self.curve_b.clear()
            
            self.curve_mono.setData(data['mono'][0], data['mono'][1])
            
        self.stats_label.setText(data['stats'])

    def update_histogram(self, img):
        # Legacy method, now unused or redirects
        pass


class VideoRecorder(QThread):
    def __init__(self):
        super().__init__()
        self.queue = queue.Queue()
        self.running = False
        self.writer = None
        self.filename = ""
        self.width = 0
        self.height = 0
        self.is_color = False
        
    def start_recording(self, filename, width, height, is_color, fps=30.0):
        self.filename = filename
        self.width = width
        self.height = height
        self.is_color = is_color
        
        # Define codec
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        self.writer = cv2.VideoWriter(self.filename, fourcc, fps, (self.width, self.height), True) # Always use color writer for compatibility
        
        self.running = True
        self.start()
        
    def add_frame(self, frame):
        if self.running:
            self.queue.put(frame)
            
    def stop_recording(self):
        self.running = False
        self.wait()
        if self.writer:
            self.writer.release()
            self.writer = None
            
    def run(self):
        while self.running or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            if self.writer:
                # Prepare frame for OpenCV (BGR)
                if len(frame.shape) == 2:
                    # Mono -> BGR
                    bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                else:
                    # RGB -> BGR
                    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Resize if needed (should match init size, but safety check)
                if bgr.shape[1] != self.width or bgr.shape[0] != self.height:
                    bgr = cv2.resize(bgr, (self.width, self.height))
                    
                self.writer.write(bgr)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("POA Camera Viewer")
        self.resize(1000, 800)
        
        self.worker = None
        self.recorder = VideoRecorder()
        self.recording = False
        self.snapshot_requested = False

        # Central Widget & Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

        # Control Panel (Left)
        controls_layout = QVBoxLayout()
        controls_group = QGroupBox("Controls")
        controls_group.setLayout(controls_layout)
        controls_group.setFixedWidth(250)
        layout.addWidget(controls_group)

        # Exposure Control
        controls_layout.addWidget(QLabel("Exposure (ms):"))
        self.exp_spin = QDoubleSpinBox()
        self.exp_spin.setRange(0.01, 60000.0) # 0.01ms to 60s
        self.exp_spin.setValue(10.0)
        self.exp_spin.setSingleStep(1.0)
        self.exp_spin.valueChanged.connect(self.update_exposure)
        controls_layout.addWidget(self.exp_spin)

        # Gain Control
        gain_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("Gain:"))
        controls_layout.addLayout(gain_layout)
        
        self.gain_spin = QSpinBox()
        self.gain_spin.setRange(0, 500) # Typical range, adjust if needed
        self.gain_spin.setValue(220)
        self.gain_spin.valueChanged.connect(self.update_gain)
        gain_layout.addWidget(self.gain_spin)
        
        self.auto_gain_check = QCheckBox("Auto")
        self.auto_gain_check.stateChanged.connect(self.toggle_auto_gain)
        gain_layout.addWidget(self.auto_gain_check)

        # Frame Rate Limit
        controls_layout.addWidget(QLabel("Frame Rate Limit (FPS):"))
        fps_layout = QHBoxLayout()
        controls_layout.addLayout(fps_layout)
        
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(0, 2000)
        self.fps_spin.setValue(10) # Default to 10 FPS
        self.fps_spin.setSpecialValueText("No Limit")
        self.fps_spin.valueChanged.connect(self.update_frame_limit)
        fps_layout.addWidget(self.fps_spin)
        
        self.fps_est_label = QLabel("Actual: 0.0 fps")
        fps_layout.addWidget(self.fps_est_label)

        # Binning Control
        controls_layout.addWidget(QLabel("Binning:"))
        self.bin_spin = QSpinBox()
        self.bin_spin.setRange(1, 4)
        self.bin_spin.setValue(2) # Default to 2
        self.bin_spin.valueChanged.connect(self.update_binning)
        controls_layout.addWidget(self.bin_spin)

        # Format Selection
        controls_layout.addWidget(QLabel("Image Format:"))
        self.format_combo = QComboBox()
        self.populate_formats()
        self.format_combo.currentTextChanged.connect(self.update_format)
        controls_layout.addWidget(self.format_combo)

        # Histogram Button
        self.btn_hist = QPushButton("Show Histogram")
        self.btn_hist.clicked.connect(self.toggle_histogram)
        controls_layout.addWidget(self.btn_hist)

        # Start/Stop Buttons
        self.btn_start = QPushButton("Start Capture")
        self.btn_start.clicked.connect(self.start_capture)
        controls_layout.addWidget(self.btn_start)

        self.btn_stop = QPushButton("Stop Capture")
        self.btn_stop.clicked.connect(self.stop_capture)
        self.btn_stop.setEnabled(False)
        controls_layout.addWidget(self.btn_stop)
        
        # Record Button
        self.btn_record = QPushButton("Record Video")
        self.btn_record.setCheckable(True)
        self.btn_record.clicked.connect(self.toggle_recording)
        self.btn_record.setEnabled(False) # Only enable when capturing
        controls_layout.addWidget(self.btn_record)
        
        # Snapshot Button
        self.btn_snapshot = QPushButton("Save Snapshot")
        self.btn_snapshot.clicked.connect(self.request_snapshot)
        self.btn_snapshot.setEnabled(False)
        controls_layout.addWidget(self.btn_snapshot)
        
        controls_layout.addStretch()

        # Image Display (Right)
        self.imv = pg.ImageView()
        # Customizing the view
        self.imv.ui.histogram.show()
        self.imv.ui.roiBtn.hide()
        self.imv.ui.menuBtn.hide()
        layout.addWidget(self.imv)

        # Status Bar
        self.status_label = QLabel("Ready")
        self.statusBar().addWidget(self.status_label)

        # Camera Worker
        self.worker = None
        
        # Histogram Window
        self.hist_window = HistogramWindow()

    def toggle_histogram(self):
        if self.hist_window.isVisible():
            self.hist_window.hide()
            self.btn_hist.setText("Show Histogram")
        else:
            self.hist_window.show()
            self.btn_hist.setText("Hide Histogram")
        
        if self.worker:
            self.worker.set_histogram_enabled(self.hist_window.isVisible())

    def start_capture(self):
        if self.worker is not None and self.worker.isRunning():
            return

        self.worker = CameraWorker()
        self.worker.image_ready.connect(self.update_image)
        self.worker.status_message.connect(self.update_status)
        self.worker.error_message.connect(self.show_error)
        self.worker.fps_updated.connect(self.update_fps_label)
        self.worker.histogram_ready.connect(self.hist_window.update_data)
        
        # Set initial values
        self.update_exposure(self.exp_spin.value())
        self.update_gain(self.gain_spin.value())
        self.update_frame_limit(self.fps_spin.value())
        self.update_binning(self.bin_spin.value())
        self.update_format(self.format_combo.currentText())
        
        self.worker.set_histogram_enabled(self.hist_window.isVisible())
        
        self.worker.start()
        
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_record.setEnabled(True)
        self.btn_snapshot.setEnabled(True)
        self.update_status("Capture started.")

    def stop_capture(self):
        if self.worker:
            self.worker.stop()
            self.worker = None
            
        if self.recording:
            self.toggle_recording() # Stop recording if active
            
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_record.setEnabled(False)
        self.btn_snapshot.setEnabled(False)
        self.update_status("Capture stopped.")

    def update_image(self, img):
        levelMode = 'mono'
        if len(img.shape) == 3:
            levelMode = 'rgba'
        self.imv.setImage(img, autoLevels=False, autoRange=False)
        
        if self.recording:
            self.recorder.add_frame(img)
            
        if self.snapshot_requested:
            self.snapshot_requested = False
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"snapshot_{timestamp}.tiff"
            # Save as TIFF using cv2
            # cv2 expects BGR for color, or grayscale
            if len(img.shape) == 3:
                # RGB -> BGR
                save_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                save_img = img
            
            cv2.imwrite(filename, save_img)
            self.update_status(f"Snapshot saved: {filename}")
            
        # if self.hist_window.isVisible():
        #    self.hist_window.update_histogram(img)

    def update_fps_label(self, fps):
        self.fps_est_label.setText(f"Actual: {fps:.1f} fps")

    def update_status(self, msg):
        self.status_label.setText(msg)

    def show_error(self, msg):
        QMessageBox.critical(self, "Camera Error", msg)
        self.stop_capture()

    def update_exposure(self, val_ms):
        if self.worker:
            us = int(val_ms * 1000)
            self.worker.set_exposure(us)

    def update_gain(self, val):
        if self.worker:
            self.worker.set_gain(val, self.auto_gain_check.isChecked())

    def toggle_recording(self):
        if self.btn_record.isChecked():
            # Start recording
            self.recording = True
            self.btn_record.setText("Stop Recording")
            self.btn_record.setStyleSheet("background-color: red; color: white;")
            
            # Generate filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.mp4"
            
            # Determine dimensions and type from current image (or mock)
            # We need the current image to know size. 
            # Ideally we get it from the worker or wait for first frame.
            # For now, let's assume the next frame in update_image will handle it? 
            # No, we need to init the recorder.
            # Let's peek at the image in the ImageView if available
            if self.imv.image is not None:
                img = self.imv.image
                h, w = img.shape[:2]
                is_color = (len(img.shape) == 3)
                self.recorder.start_recording(filename, w, h, is_color)
                self.update_status(f"Recording to {filename}...")
            else:
                # Should not happen if capture is running and we have frames
                self.recording = False
                self.btn_record.setChecked(False)
                self.update_status("No image to record.")
                
        else:
            # Stop recording
            self.recording = False
            self.btn_record.setText("Record Video")
            self.btn_record.setStyleSheet("")
            self.recorder.stop_recording()
            self.update_status(f"Recording saved: {self.recorder.filename}")

    def request_snapshot(self):
        self.snapshot_requested = True
        self.update_status("Waiting for next frame to save snapshot...")

    def toggle_auto_gain(self, state):
        is_auto = (state == Qt.Checked)
        self.gain_spin.setEnabled(not is_auto)
        if self.worker:
            self.worker.set_gain(self.gain_spin.value(), is_auto)

    def update_frame_limit(self, val):
        if self.worker:
            self.worker.set_frame_limit(val)

    def update_binning(self, val):
        if self.worker:
            self.worker.set_binning(val)

    def populate_formats(self):
        # In a real app, we should query the camera for supported formats.
        # For now, we list the common ones from the Enum.
        # We could try to get properties if a camera is connected, but we do this before start.
        # Let's try to peek at camera 0 if available.
        formats = ["POA_RAW8", "POA_RAW16", "POA_RGB24", "POA_MONO8"]
        
        # Try to get actual supported formats from camera 0
        try:
            count = pyPOACamera.GetCameraCount()
            if count > 0:
                err, props = pyPOACamera.GetCameraProperties(0)
                if err == pyPOACamera.POAErrors.POA_OK:
                    supported = props.imgFormats
                    formats = [fmt.name for fmt in supported]
        except Exception:
            pass # Fallback to default list if library fails or no camera

        self.format_combo.addItems(formats)

    def update_format(self, text):
        if self.worker:
            self.worker.set_format(text)

    def closeEvent(self, event):
        self.stop_capture()
        self.hist_window.close()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
