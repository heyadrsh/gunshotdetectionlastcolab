import os
import time
import random
import threading
import sounddevice as sd
import soundfile as sf
from pathlib import Path

class GunShotTester:
    def __init__(self, gunshot_dir="dataset/gunshots"):
        """Initialize gunshot tester
        
        Args:
            gunshot_dir: Directory containing gunshot audio files
        """
        self.gunshot_dir = Path(gunshot_dir)
        self.is_running = False
        self.play_thread = None
        
        # Get list of gunshot files
        self.gunshot_files = []
        for gun_folder in self.gunshot_dir.iterdir():
            if gun_folder.is_dir():
                self.gunshot_files.extend(list(gun_folder.glob("*.wav")))
        
        print(f"Found {len(self.gunshot_files)} gunshot audio files")
        
    def play_audio(self, audio_file):
        """Play an audio file through the system's audio output"""
        try:
            # Load audio file
            audio_data, sample_rate = sf.read(audio_file)
            
            # Play audio
            sd.play(audio_data, sample_rate)
            sd.wait()  # Wait until audio finishes playing
            
        except Exception as e:
            print(f"Error playing audio: {str(e)}")
            
    def test_loop(self):
        """Main testing loop that plays random gunshot sounds"""
        print("\nStarting gunshot playback test...")
        print("Will play random gunshot sounds every 10-20 seconds")
        print("Press Ctrl+C to stop")
        
        while self.is_running:
            try:
                # Random delay between gunshots (10-20 seconds)
                delay = random.uniform(10, 20)
                time.sleep(delay)
                
                # Select random gunshot file
                gunshot_file = random.choice(self.gunshot_files)
                print(f"\nPlaying gunshot from: {gunshot_file.parent.name}")
                
                # Play the gunshot
                self.play_audio(gunshot_file)
                
            except Exception as e:
                print(f"Error in test loop: {str(e)}")
                
    def start(self):
        """Start the testing process"""
        try:
            self.is_running = True
            
            # Start playback thread
            self.play_thread = threading.Thread(target=self.test_loop)
            self.play_thread.start()
            
            # Keep main thread alive
            while self.is_running:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nStopping test...")
        finally:
            self.stop()
            
    def stop(self):
        """Stop the testing process"""
        self.is_running = False
        if self.play_thread:
            self.play_thread.join()
        print("Testing stopped.")

def main():
    """Main function to run gunshot testing"""
    try:
        # Create and start tester
        tester = GunShotTester()
        tester.start()
    except KeyboardInterrupt:
        print("\nTest terminated by user")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 