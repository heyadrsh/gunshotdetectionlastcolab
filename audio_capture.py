import pyaudio
import numpy as np
import wave
from datetime import datetime
import os

class AudioCapture:
    def __init__(self, 
                 rate=44100,
                 chunk_size=1024,
                 channels=1,
                 record_seconds=1):
        self.rate = rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.record_seconds = record_seconds
        self.format = pyaudio.paFloat32
        self.p = pyaudio.PyAudio()
        
    def setup_stream(self):
        """Setup audio stream for continuous capture"""
        stream = self.p.open(format=self.format,
                           channels=self.channels,
                           rate=self.rate,
                           input=True,
                           frames_per_buffer=self.chunk_size)
        return stream
    
    def record_audio(self, save=False, filename=None):
        """Record audio for specified duration"""
        stream = self.setup_stream()
        frames = []
        
        # Calculate number of chunks to read
        chunks_to_read = int(self.rate / self.chunk_size * self.record_seconds)
        
        print("* recording")
        
        # Record audio
        for i in range(chunks_to_read):
            data = stream.read(self.chunk_size)
            frames.append(np.frombuffer(data, dtype=np.float32))
            
        print("* done recording")
        
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        
        # Combine all frames into a single numpy array
        audio_data = np.concatenate(frames)
        
        if save:
            if filename is None:
                filename = f"recorded_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            
            # Save as WAV file
            wf = wave.open(filename, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.p.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(audio_data.tobytes())
            wf.close()
            
        return audio_data
    
    def start_continuous_capture(self, callback):
        """Start continuous audio capture with callback for processing"""
        stream = self.setup_stream()
        
        try:
            while True:
                data = stream.read(self.chunk_size)
                audio_chunk = np.frombuffer(data, dtype=np.float32)
                callback(audio_chunk)
                
        except KeyboardInterrupt:
            print("* Stopping recording")
            stream.stop_stream()
            stream.close()
            
    def __del__(self):
        """Cleanup PyAudio"""
        self.p.terminate()

if __name__ == "__main__":
    # Test audio capture
    capture = AudioCapture()
    audio_data = capture.record_audio(save=True)
    print(f"Recorded audio shape: {audio_data.shape}") 