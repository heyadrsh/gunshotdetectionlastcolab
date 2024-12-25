import numpy as np
import librosa
from scipy.spatial.distance import pdist, squareform
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import soundfile as sf
import os

class GunShotFeatureExtractor:
    def __init__(self, 
                 sample_rate: int = 44100,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 n_mels: int = 128,
                 n_mfcc: int = 20,
                 window_size: float = 1.0):  # 1-second window
        """
        Initialize feature extractor with parameters optimized for gunshot detection
        
        Args:
            sample_rate: Audio sample rate (default: 44100 Hz)
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
            n_mels: Number of mel bands
            n_mfcc: Number of MFCC coefficients
            window_size: Analysis window size in seconds
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.window_size = window_size
        self.target_shape = (224, 224)  # Standard input size for CNNs
        
    def extract_features(self, audio_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract all features from audio data
        
        Args:
            audio_data: Audio signal array
            
        Returns:
            Dictionary containing spectrogram, MFCC, and similarity matrix features
        """
        # Ensure audio is the right length
        target_length = int(self.window_size * self.sample_rate)
        if len(audio_data) > target_length:
            audio_data = audio_data[:target_length]
        elif len(audio_data) < target_length:
            # Pad with zeros if too short
            audio_data = np.pad(audio_data, (0, target_length - len(audio_data)))
        
        # Extract individual features
        spectrogram = self.get_spectrogram(audio_data)
        mfcc = self.get_mfcc(audio_data)
        similarity = self.get_similarity_matrix(audio_data)
        
        # Resize all features to target shape
        spec_resized = self._resize_feature(spectrogram)
        mfcc_resized = self._resize_feature(mfcc)
        sim_resized = self._resize_feature(similarity)
        
        # Combine features into RGB-like channels
        combined_features = np.stack([spec_resized, mfcc_resized, sim_resized], axis=-1)
        
        return {
            'spectrogram': spec_resized,
            'mfcc': mfcc_resized,
            'similarity': sim_resized,
            'combined': combined_features
        }
    
    def get_spectrogram(self, audio_data: np.ndarray) -> np.ndarray:
        """Generate optimized spectrogram for gunshot detection"""
        # Apply short-time Fourier transform
        D = librosa.stft(audio_data, 
                        n_fft=self.n_fft, 
                        hop_length=self.hop_length,
                        window='hann')
        
        # Convert to mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            S=np.abs(D)**2,
            sr=self.sample_rate,
            n_mels=self.n_mels
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        log_mel_spec = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min())
        
        return log_mel_spec
    
    def get_mfcc(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract MFCC features optimized for gunshot characteristics"""
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(
            y=audio_data,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Add delta and delta-delta features
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        
        # Combine all MFCC features
        mfcc_features = np.concatenate([mfcc, delta_mfcc, delta2_mfcc])
        
        # Normalize
        mfcc_features = (mfcc_features - mfcc_features.min()) / (mfcc_features.max() - mfcc_features.min())
        
        return mfcc_features
    
    def get_similarity_matrix(self, audio_data: np.ndarray) -> np.ndarray:
        """Generate self-similarity matrix for temporal pattern analysis"""
        # Extract MFCC features for similarity calculation
        mfcc = librosa.feature.mfcc(
            y=audio_data,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Calculate pairwise distances
        distances = pdist(mfcc.T, metric='euclidean')
        
        # Convert to square matrix
        similarity_matrix = squareform(distances)
        
        # Normalize
        similarity_matrix = (similarity_matrix - similarity_matrix.min()) / \
                          (similarity_matrix.max() - similarity_matrix.min())
        
        return similarity_matrix
    
    def _resize_feature(self, feature: np.ndarray) -> np.ndarray:
        """Resize feature matrix to target shape"""
        from scipy.ndimage import zoom
        
        # Calculate zoom factors
        zoom_factors = (self.target_shape[0] / feature.shape[0],
                       self.target_shape[1] / feature.shape[1])
        
        # Apply zoom
        resized = zoom(feature, zoom_factors, order=1)
        
        return resized
    
    def visualize_features(self, features: Dict[str, np.ndarray], save_path: Optional[str] = None):
        """Visualize extracted features"""
        plt.figure(figsize=(15, 5))
        
        # Plot spectrogram
        plt.subplot(131)
        plt.imshow(features['spectrogram'])
        plt.title('Mel Spectrogram')
        plt.colorbar(format='%+2.0f dB')
        
        # Plot MFCC
        plt.subplot(132)
        plt.imshow(features['mfcc'])
        plt.title('MFCC Features')
        plt.colorbar()
        
        # Plot similarity matrix
        plt.subplot(133)
        plt.imshow(features['similarity'])
        plt.title('Similarity Matrix')
        plt.colorbar()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

if __name__ == "__main__":
    # Test the feature extractor
    from data_loader import GunShotDataLoader
    
    # Initialize loader and extractor
    loader = GunShotDataLoader()
    extractor = GunShotFeatureExtractor()
    
    # Try to load a test file directly
    test_path = "dataset/gunshots/glock_17_9mm_caliber"
    if os.path.exists(test_path):
        # Find first WAV file
        wav_files = [f for f in os.listdir(test_path) if f.endswith('.wav')]
        if wav_files:
            test_file = os.path.join(test_path, wav_files[0])
            print(f"Loading test file: {test_file}")
            
            # Load audio file
            audio_data, sr = sf.read(test_file)
            
            # Extract features
            features = extractor.extract_features(audio_data)
            
            # Visualize features
            extractor.visualize_features(features, "gunshot_features.png")
            print("Features extracted and visualized successfully!")
            print(f"Combined feature shape: {features['combined'].shape}")
        else:
            print("No WAV files found in test directory")
    else:
        print(f"Test directory not found: {test_path}") 