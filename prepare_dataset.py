import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
from feature_extraction import GunShotFeatureExtractor
import h5py
import librosa

class DatasetPreparator:
    def __init__(self, 
                 gunshot_path: str = "dataset/gunshots",
                 urbansound_path: str = "dataset/urbansound8k",
                 output_path: str = "dataset/processed",
                 max_audio_length: float = 4.0):  # Maximum audio length in seconds
        """
        Prepare dataset for training by processing gunshot and background sounds
        
        Args:
            gunshot_path: Path to gunshot dataset
            urbansound_path: Path to UrbanSound8K dataset
            output_path: Where to save processed features
            max_audio_length: Maximum audio length in seconds to process
        """
        self.gunshot_path = Path(gunshot_path)
        self.urbansound_path = Path(urbansound_path)
        self.output_path = Path(output_path)
        self.feature_extractor = GunShotFeatureExtractor()
        self.max_audio_length = max_audio_length
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_path, exist_ok=True)
        
    def load_and_trim_audio(self, audio_path: Path, sr: int = 22050) -> tuple:
        """Load and trim audio to max length"""
        try:
            # Load audio with librosa (resampling if needed)
            audio_data, actual_sr = librosa.load(audio_path, sr=sr, duration=self.max_audio_length)
            
            # Ensure minimum length (pad if needed)
            min_samples = int(0.1 * sr)  # Minimum 100ms
            if len(audio_data) < min_samples:
                return None, None
                
            return audio_data, actual_sr
        except Exception as e:
            print(f"Error loading {audio_path}: {str(e)}")
            return None, None
        
    def process_gunshots(self):
        """Process all gunshot audio files"""
        print("Processing gunshot sounds...")
        features = []
        labels = []
        
        # Process each gun type folder
        for gun_folder in self.gunshot_path.iterdir():
            if not gun_folder.is_dir():
                continue
                
            print(f"Processing {gun_folder.name}...")
            
            # Process each WAV file
            for wav_file in tqdm(list(gun_folder.glob("*.wav"))):
                try:
                    # Load and trim audio
                    audio_data, sr = self.load_and_trim_audio(wav_file)
                    if audio_data is None:
                        continue
                    
                    # Extract features
                    feature_dict = self.feature_extractor.extract_features(audio_data)
                    features.append(feature_dict['combined'])
                    labels.append(1)  # 1 for gunshot
                    
                except Exception as e:
                    print(f"Error processing {wav_file}: {str(e)}")
                    
        # Convert to numpy arrays with proper shape
        features = np.array(features)
        labels = np.array(labels)
        
        print(f"Processed {len(features)} gunshot samples")
        print(f"Feature shape: {features.shape}")
        
        return features, labels
    
    def process_background(self, max_samples: int = None):
        """Process background sounds from UrbanSound8K"""
        print("\nProcessing background sounds...")
        features = []
        labels = []
        
        # Read UrbanSound8K metadata
        metadata_path = self.urbansound_path / "UrbanSound8K.csv"
        if not metadata_path.exists():
            print(f"UrbanSound8K metadata not found at {metadata_path}")
            return np.array([]), np.array([])
            
        df = pd.read_csv(metadata_path)
        
        # Process each audio file
        processed = 0
        skipped = 0
        for _, row in tqdm(df.iterrows(), total=len(df)):
            if max_samples and processed >= max_samples:
                break
                
            try:
                # Construct file path
                audio_path = self.urbansound_path / f"fold{row['fold']}" / row['slice_file_name']
                
                # Load and trim audio
                audio_data, sr = self.load_and_trim_audio(audio_path)
                if audio_data is None:
                    skipped += 1
                    continue
                
                # Extract features
                feature_dict = self.feature_extractor.extract_features(audio_data)
                features.append(feature_dict['combined'])
                labels.append(0)  # 0 for background
                
                processed += 1
                
            except Exception as e:
                print(f"Error processing {audio_path}: {str(e)}")
                skipped += 1
                
        # Convert to numpy arrays with proper shape
        features = np.array(features) if features else np.array([])
        labels = np.array(labels)
        
        print(f"Processed {len(features)} background samples")
        print(f"Skipped {skipped} samples")
        if len(features) > 0:
            print(f"Feature shape: {features.shape}")
        
        return features, labels
    
    def prepare_dataset(self, train_split: float = 0.8):
        """Prepare complete dataset with train/test split"""
        # Process gunshots
        gunshot_features, gunshot_labels = self.process_gunshots()
        
        if len(gunshot_features) == 0:
            print("No gunshot samples processed. Aborting.")
            return
            
        # Process background sounds - match the number of gunshot samples
        n_gunshots = len(gunshot_features)
        background_features, background_labels = self.process_background(max_samples=n_gunshots)
        
        if len(background_features) == 0:
            print("No background samples found. Cannot proceed without background sounds.")
            return
        elif len(background_features) < n_gunshots:
            print(f"Warning: Only found {len(background_features)} background samples.")
            print("Reducing gunshot samples to match...")
            # Randomly select matching number of gunshot samples
            indices = np.random.choice(len(gunshot_features), size=len(background_features), replace=False)
            gunshot_features = gunshot_features[indices]
            gunshot_labels = gunshot_labels[indices]
            
        # Combine features and labels
        features = np.concatenate([gunshot_features, background_features], axis=0)
        labels = np.concatenate([gunshot_labels, background_labels])
        
        # Shuffle data
        indices = np.arange(len(features))
        np.random.shuffle(indices)
        features = features[indices]
        labels = labels[indices]
        
        # Split into train and test
        split_idx = int(len(features) * train_split)
        train_features = features[:split_idx]
        train_labels = labels[:split_idx]
        test_features = features[split_idx:]
        test_labels = labels[split_idx:]
        
        # Print class distribution
        print("\nClass distribution:")
        print(f"Total samples: {len(features)}")
        print(f"Gunshots: {np.sum(labels == 1)} ({np.sum(labels == 1)/len(labels)*100:.1f}%)")
        print(f"Background: {np.sum(labels == 0)} ({np.sum(labels == 0)/len(labels)*100:.1f}%)")
        
        # Save processed dataset
        print("\nSaving processed dataset...")
        with h5py.File(self.output_path / "gunshot_detection_dataset.h5", 'w') as f:
            # Training data
            f.create_dataset('train_features', data=train_features)
            f.create_dataset('train_labels', data=train_labels)
            
            # Test data
            f.create_dataset('test_features', data=test_features)
            f.create_dataset('test_labels', data=test_labels)
            
        print(f"\nDataset prepared successfully!")
        print(f"Training samples: {len(train_features)}")
        print(f"Test samples: {len(test_features)}")
        print(f"\nDataset saved to: {self.output_path / 'gunshot_detection_dataset.h5'}")

if __name__ == "__main__":
    preparator = DatasetPreparator(max_audio_length=4.0)  # Limit audio to 4 seconds
    preparator.prepare_dataset()  # Will automatically balance gunshot and background samples