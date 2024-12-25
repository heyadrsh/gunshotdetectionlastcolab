import os
import pandas as pd
import numpy as np
import librosa
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

class GunShotDataLoader:
    def __init__(self, dataset_path: str = "dataset/gunshots"):
        self.dataset_path = Path(dataset_path)
        self.gun_types = {
            "glock_17_9mm_caliber": "Glock 17 9mm",
            "38s&ws_dot38_caliber": "S&W .38 Special",
            "remington_870_12_gauge": "Remington 870",
            "ruger_ar_556_dot223_caliber": "Ruger AR-556"
        }
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict:
        """Load and organize dataset metadata"""
        metadata = {
            "gun_types": {},
            "total_samples": 0,
            "channel_info": {},
            "sample_rate": 44100  # Standard sample rate for the dataset
        }
        
        # Count samples for each gun type
        for gun_folder in self.gun_types.keys():
            gun_path = self.dataset_path / gun_folder
            if not gun_path.exists():
                continue
                
            wav_files = list(gun_path.glob("*.wav"))
            metadata["gun_types"][self.gun_types[gun_folder]] = {
                "total_files": len(wav_files),
                "multichannel_recordings": len([f for f in wav_files if "chan" in f.name]),
                "mean_recordings": len([f for f in wav_files if "mean" in f.name]),
                "single_recordings": len([f for f in wav_files if "mean" not in f.name and "chan" not in f.name])
            }
            metadata["total_samples"] += len(wav_files)
            
        return metadata
    
    def load_audio_file(self, file_path: str, duration: Optional[float] = None) -> Tuple[np.ndarray, int]:
        """Load a single audio file"""
        try:
            audio_data, sr = librosa.load(file_path, sr=None, duration=duration)
            return audio_data, sr
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            return np.array([]), 0
    
    def load_multichannel_recording(self, recording_id: str, gun_type: str) -> Dict:
        """Load all channels for a specific recording ID"""
        gun_folder = [k for k, v in self.gun_types.items() if v == gun_type][0]
        base_path = self.dataset_path / gun_folder
        
        result = {
            "channels": {},
            "mean": None,
            "sample_rate": None,
            "gun_type": gun_type
        }
        
        # Load individual channels
        for i in range(7):  # 7 channels (0-6)
            file_path = base_path / f"{recording_id}_chan{i}_v0.wav"
            if file_path.exists():
                audio_data, sr = self.load_audio_file(str(file_path))
                if len(audio_data) > 0:
                    result["channels"][i] = audio_data
                    result["sample_rate"] = sr
        
        # Load mean recording if available
        mean_path = base_path / f"{recording_id}_mean_v0.wav"
        if mean_path.exists():
            mean_audio, sr = self.load_audio_file(str(mean_path))
            if len(mean_audio) > 0:
                result["mean"] = mean_audio
                
        return result
    
    def get_recording_ids(self, gun_type: Optional[str] = None) -> List[str]:
        """Get all unique recording IDs, optionally filtered by gun type"""
        recording_ids = set()
        
        if gun_type:
            gun_folder = [k for k, v in self.gun_types.items() if v == gun_type][0]
            search_folders = [gun_folder]
        else:
            search_folders = self.gun_types.keys()
            
        for folder in search_folders:
            folder_path = self.dataset_path / folder
            if not folder_path.exists():
                continue
                
            for file_path in folder_path.glob("*.wav"):
                # Extract the UUID part of the filename
                recording_id = file_path.stem.split('_')[0]
                recording_ids.add(recording_id)
                
        return list(recording_ids)
    
    def get_dataset_info(self) -> str:
        """Get formatted information about the dataset"""
        info = ["Gunshot Dataset Information:"]
        info.append(f"Total samples: {self.metadata['total_samples']}")
        info.append("\nGun Types:")
        
        for gun_type, stats in self.metadata["gun_types"].items():
            info.append(f"\n{gun_type}:")
            info.append(f"  - Total files: {stats['total_files']}")
            info.append(f"  - Multichannel recordings: {stats['multichannel_recordings']}")
            info.append(f"  - Mean recordings: {stats['mean_recordings']}")
            info.append(f"  - Single recordings: {stats['single_recordings']}")
            
        return "\n".join(info)
    
    def save_metadata(self, output_file: str = "dataset_metadata.json"):
        """Save dataset metadata to a JSON file"""
        with open(output_file, 'w') as f:
            json.dump(self.metadata, f, indent=4)

if __name__ == "__main__":
    # Test the data loader
    loader = GunShotDataLoader()
    
    # Print dataset information
    print(loader.get_dataset_info())
    
    # Save metadata
    loader.save_metadata()
    
    # Test loading a multichannel recording
    recording_ids = loader.get_recording_ids("Glock 17 9mm")
    if recording_ids:
        test_id = recording_ids[0]
        print(f"\nTesting multichannel loading for recording {test_id}")
        data = loader.load_multichannel_recording(test_id, "Glock 17 9mm")
        print(f"Loaded {len(data['channels'])} channels")
        if data['mean'] is not None:
            print("Mean recording loaded successfully") 