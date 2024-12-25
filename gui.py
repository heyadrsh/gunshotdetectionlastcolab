import sys
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QProgressBar)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
import sounddevice as sd
from datetime import datetime
import os
from feature_extraction import GunShotFeatureExtractor
import tensorflow as tf

class GunShotDetectorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gunshot Detection System")
        self.setGeometry(100, 100, 800, 600)
        
        # Gun type mapping
        self.gun_types = {
            0: "No Gunshot",
            1: "Glock 17 9mm",
            2: "S&W .38 Special",
            3: "Remington 870",
            4: "Ruger AR-556"
        }
        
        # Initialize detector components
        self.setup_detector()
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Add title
        title = QLabel("Real-time Gunshot Detection")
        title.setFont(QFont('Arial', 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Add audio visualization
        self.setup_audio_plot(layout)
        
        # Add detection status
        self.setup_status_display(layout)
        
        # Add controls
        self.setup_controls(layout)
        
        # Setup timers
        self.setup_timers()
        
        # Initialize state
        self.is_detecting = False
        self.detection_threshold = 0.95  # 95% confidence threshold
        
    def setup_detector(self):
        """Initialize detection components"""
        self.sample_rate = 44100
        self.chunk_duration = 1.0  # seconds
        self.chunk_samples = int(self.sample_rate * self.chunk_duration)
        self.feature_extractor = GunShotFeatureExtractor(sample_rate=self.sample_rate)
        
        # Load model
        print("Loading model...")
        try:
            self.model = GunShotDetector(num_classes=5)
            self.model = self.model.load_model("checkpoints/model_best.keras")
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.model = None
            
        # Create output directory
        os.makedirs("detected_gunshots", exist_ok=True)
        
    def setup_audio_plot(self, layout):
        """Setup real-time audio waveform plot"""
        # Create plot widget
        self.audio_plot = pg.PlotWidget()
        self.audio_plot.setBackground('w')
        self.audio_plot.setTitle("Audio Waveform")
        self.audio_plot.setLabel('left', 'Amplitude')
        self.audio_plot.setLabel('bottom', 'Sample')
        self.audio_plot.showGrid(x=True, y=True)
        
        # Create plot line
        self.plot_line = self.audio_plot.plot(pen='b')
        
        # Add to layout
        layout.addWidget(self.audio_plot)
        
    def setup_status_display(self, layout):
        """Setup detection status display"""
        status_layout = QVBoxLayout()
        
        # Detection status
        self.status_label = QLabel("Status: Not Detecting")
        self.status_label.setFont(QFont('Arial', 12))
        self.status_label.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.status_label)
        
        # Gun type prediction
        self.gun_type_label = QLabel("Detected Gun: None")
        self.gun_type_label.setFont(QFont('Arial', 12))
        self.gun_type_label.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.gun_type_label)
        
        # Confidence bars layout
        confidence_layout = QVBoxLayout()
        self.confidence_bars = {}
        
        for gun_id, gun_name in self.gun_types.items():
            # Create horizontal layout for each gun type
            gun_layout = QHBoxLayout()
            
            # Add label
            label = QLabel(f"{gun_name}:")
            label.setMinimumWidth(120)
            gun_layout.addWidget(label)
            
            # Add progress bar
            progress = QProgressBar()
            progress.setRange(0, 100)
            gun_layout.addWidget(progress)
            
            # Store progress bar
            self.confidence_bars[gun_id] = progress
            
            # Add to confidence layout
            confidence_layout.addLayout(gun_layout)
        
        status_layout.addLayout(confidence_layout)
        
        # Last detection
        self.last_detection_label = QLabel("Last Detection: None")
        self.last_detection_label.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.last_detection_label)
        
        layout.addLayout(status_layout)
        
    def setup_controls(self, layout):
        """Setup control buttons"""
        controls_layout = QHBoxLayout()
        
        # Start button
        self.start_button = QPushButton("Start Detection")
        self.start_button.clicked.connect(self.toggle_detection)
        controls_layout.addWidget(self.start_button)
        
        # Test button
        self.test_button = QPushButton("Play Test Sound")
        self.test_button.clicked.connect(self.play_test_sound)
        controls_layout.addWidget(self.test_button)
        
        layout.addLayout(controls_layout)
        
    def setup_timers(self):
        """Setup update timers"""
        # Timer for audio plot updates
        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self.update_plot)
        self.plot_timer.start(50)  # Update every 50ms
        
        # Audio buffer for plotting
        self.audio_buffer = np.zeros(self.chunk_samples)
        
    def toggle_detection(self):
        """Start/stop detection"""
        if not self.is_detecting:
            if self.model is None:
                self.status_label.setText("Status: Error - Model not loaded!")
                return
                
            try:
                # Start audio stream
                self.audio_stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    callback=self.audio_callback,
                    blocksize=self.chunk_samples
                )
                self.audio_stream.start()
                
                self.is_detecting = True
                self.start_button.setText("Stop Detection")
                self.status_label.setText("Status: Detecting")
                
            except Exception as e:
                self.status_label.setText(f"Status: Error - {str(e)}")
                
        else:
            # Stop detection
            if hasattr(self, 'audio_stream'):
                self.audio_stream.stop()
                self.audio_stream.close()
            
            self.is_detecting = False
            self.start_button.setText("Start Detection")
            self.status_label.setText("Status: Not Detecting")
            self.confidence_bar.setValue(0)
            
    def audio_callback(self, indata, frames, time, status):
        """Process audio input"""
        if status:
            print(f"Status: {status}")
            
        # Get audio data
        audio_chunk = indata[:, 0]
        self.audio_buffer = audio_chunk
        
        if self.is_detecting:
            self.process_audio(audio_chunk)
            
    def process_audio(self, audio_chunk):
        """Process audio chunk for gunshot detection"""
        try:
            # Calculate audio level
            audio_level = np.abs(audio_chunk).mean()
            
            # Extract features
            features = self.feature_extractor.extract_features(audio_chunk)
            
            # Make prediction
            predictions = self.model.predict(
                np.expand_dims(features['combined'], axis=0), 
                verbose=0
            )[0]
            
            # Update confidence bars
            for gun_id, confidence in enumerate(predictions):
                self.confidence_bars[gun_id].setValue(int(confidence * 100))
            
            # Get predicted class and confidence
            predicted_class = np.argmax(predictions)
            max_confidence = predictions[predicted_class]
            
            # Check for gunshot detection with threshold
            if predicted_class > 0 and max_confidence > self.detection_threshold and audio_level > 0.01:
                self.handle_detection(audio_chunk, predicted_class, max_confidence)
                
        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            
    def handle_detection(self, audio_chunk, gun_type, confidence):
        """Handle gunshot detection"""
        # Update last detection time
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.last_detection_label.setText(f"Last Detection: {timestamp}")
        
        # Update gun type
        gun_name = self.gun_types[gun_type]
        self.gun_type_label.setText(f"Detected Gun: {gun_name}")
        
        # Save audio clip
        filename = f"detected_gunshots/gunshot_{gun_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        import soundfile as sf
        sf.write(filename, audio_chunk, self.sample_rate)
        
        # Update status
        self.status_label.setText(f"Status: GUNSHOT DETECTED! ({confidence*100:.1f}% confidence)")
        
    def update_plot(self):
        """Update audio waveform plot"""
        self.plot_line.setData(self.audio_buffer)
        
    def play_test_sound(self):
        """Play a test gunshot sound"""
        # TODO: Implement test sound playback
        pass
        
    def closeEvent(self, event):
        """Handle application closure"""
        if hasattr(self, 'audio_stream'):
            self.audio_stream.stop()
            self.audio_stream.close()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = GunShotDetectorGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 