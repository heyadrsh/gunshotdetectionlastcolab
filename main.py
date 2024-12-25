import numpy as np
import sounddevice as sd
import soundfile as sf
from datetime import datetime
import os
from feature_extraction import GunShotFeatureExtractor
import tensorflow as tf
from pathlib import Path

class RealTimeGunShotDetector:
    def __init__(self, model_path="checkpoints/model_best.keras"):
        """Initialize real-time gunshot detector"""
        self.sample_rate = 44100
        self.chunk_duration = 1.0  # seconds
        self.chunk_samples = int(self.sample_rate * self.chunk_duration)
        self.confidence_threshold = 0.95  # Higher confidence threshold
        self.min_audio_level = 0.01  # Minimum audio level to consider
        self.cooldown = 2.0  # Seconds to wait between detections
        self.last_detection_time = None
        
        # Create output directory
        os.makedirs("detected_gunshots", exist_ok=True)
        
        # Initialize feature extractor
        self.feature_extractor = GunShotFeatureExtractor(sample_rate=self.sample_rate)
        
        # Load model
        print("Loading model...")
        self.model = tf.keras.models.load_model(model_path)
        
        # List available audio devices
        print("\nAvailable audio devices:")
        for i, device in enumerate(sd.query_devices()):
            print(f"{i}: {device['name']} (inputs: {device['max_input_channels']})")
        
        # Get default input device
        self.device = sd.default.device[0]
        print(f"\nDefault input device: {sd.query_devices(self.device)['name']}")
        
        # Test microphone access
        print("\nTesting microphone access...")
        try:
            with sd.InputStream(samplerate=self.sample_rate, channels=1,
                              callback=lambda *args: None, device=self.device):
                pass
            print("✓ Microphone access successful!")
        except Exception as e:
            print(f"Error accessing microphone: {e}")
            raise
    
    def process_audio_chunk(self, audio_chunk):
        """Process a chunk of audio data and detect gunshots"""
        try:
            # Calculate audio level
            audio_level = np.abs(audio_chunk).mean()
            
            # Skip processing if audio level is too low
            if audio_level < self.min_audio_level:
                print(f"Audio level: {audio_level:.4f} (too low)", end="\r")
                return
            
            # Check cooldown period
            current_time = datetime.now()
            if self.last_detection_time is not None:
                time_since_last = (current_time - self.last_detection_time).total_seconds()
                if time_since_last < self.cooldown:
                    return
            
            # Ensure audio chunk is the right shape
            if len(audio_chunk) < self.chunk_samples:
                audio_chunk = np.pad(audio_chunk, (0, self.chunk_samples - len(audio_chunk)))
            elif len(audio_chunk) > self.chunk_samples:
                audio_chunk = audio_chunk[:self.chunk_samples]
            
            # Extract features
            features = self.feature_extractor.extract_features(audio_chunk)
            
            # Make prediction
            prediction = self.model.predict(np.expand_dims(features['combined'], axis=0), verbose=0)
            confidence = prediction[0][0]  # Probability of gunshot class
            
            # Print debug info
            print(f"Audio: {audio_level:.4f} | Confidence: {confidence:.4f}", end="\r")
            
            # Detect gunshot if confidence exceeds threshold and audio level is significant
            if confidence > self.confidence_threshold and audio_level > self.min_audio_level:
                self.last_detection_time = current_time
                print(f"\nPotential gunshot detected!")
                print(f"Audio level: {audio_level:.4f}")
                print(f"Confidence: {confidence:.4f}")
                
                # Save the audio clip
                timestamp = current_time.strftime("%Y%m%d_%H%M%S")
                filename = f"detected_gunshots/gunshot_{timestamp}.wav"
                sf.write(filename, audio_chunk, self.sample_rate)
                print(f"Saved audio clip to {filename}\n")
                
        except Exception as e:
            print(f"\nError processing audio chunk: {str(e)}")
    
    def audio_callback(self, indata, frames, time, status):
        """Callback function to process audio input"""
        if status:
            print(f"Status: {status}")
        
        # Process the audio chunk
        audio_chunk = indata[:, 0]  # Take first channel if stereo
        self.process_audio_chunk(audio_chunk)
    
    def start_detection(self):
        """Start real-time gunshot detection"""
        print("\nReal-time gunshot detection started!")
        print(f"Minimum audio level: {self.min_audio_level}")
        print(f"Confidence threshold: {self.confidence_threshold}")
        print("Listening for gunshots... Press Ctrl+C to stop.\n")
        
        try:
            with sd.InputStream(samplerate=self.sample_rate,
                              channels=1,
                              callback=self.audio_callback,
                              blocksize=self.chunk_samples,
                              device=self.device):
                print("Audio input levels (should change with sound):")
                while True:
                    sd.sleep(1000)  # Sleep for 1 second
                    
        except KeyboardInterrupt:
            print("\nStopping detection...")
        except Exception as e:
            print(f"\nError: {str(e)}")
        finally:
            print("Detection stopped.")

if __name__ == "__main__":
    detector = RealTimeGunShotDetector()
    detector.start_detection() 