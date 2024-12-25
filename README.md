# Gunshot Detection System

A real-time gunshot detection system using deep learning and audio processing.

## Overview

This system uses a deep learning model to detect gunshots in real-time audio streams. It processes audio input through feature extraction and uses a trained convolutional neural network to classify sounds as either gunshots or background noise.

## Features

- Real-time gunshot detection using microphone input
- Audio feature extraction using spectrograms
- Deep learning model based on ResNet50V2 architecture
- Support for multiple gun types
- Background noise filtering

## Dataset Sources

The model is trained on two primary datasets:

1. **Gunshot Audio Dataset**: [Edge-Collected Gunshot Audio Dataset](https://zenodo.org/records/7004819)
   - Multi-firearm, multi-orientation gunshot audio dataset
   - Collected using edge devices in outdoor firearm range settings
   - Suitable for gunshot detection and firearm classification

2. **Background Noise Dataset**: [UrbanSound8K Dataset](https://www.kaggle.com/datasets/chrisfilo/urbansound8k)
   - Contains 8732 labeled sound excerpts of urban sounds
   - 10 classes of urban noise
   - Used for training the model to distinguish between gunshots and common background sounds

## Directory Structure

```
.
├── dataset/
│   ├── gunshots/          # Gunshot audio files
│   ├── urbansound8k/      # Background noise audio files
│   └── processed/         # Processed features in HDF5 format
├── checkpoints/           # Model checkpoints during training
├── detected_gunshots/     # Saved clips of detected gunshots
└── [other project files]
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/heyadrsh/gunshotdetectionlast.git
cd gunshotdetectionlast
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the datasets:
   - Download gunshot audio files from [Zenodo](https://zenodo.org/records/7004819)
   - Download UrbanSound8K dataset from [Kaggle](https://www.kaggle.com/datasets/chrisfilo/urbansound8k)
   - Place the files in their respective directories under `dataset/`

## Usage

1. Prepare the dataset:
```bash
python prepare_dataset.py
```

2. Train the model:
```bash
python train.py
```

3. Run real-time detection:
```bash
python main.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Training on Google Colab

For users with limited computational resources, we provide a Google Colab notebook for training:

1. Open `gunshot_detection_colab.ipynb` in Google Colab
2. Mount your Google Drive
3. Create a folder in your Google Drive named `gunshot_dataset`
4. Upload your processed dataset (h5 file) to the `gunshot_dataset` folder
5. Run all cells in sequence
6. The trained model will be saved to `/content/drive/MyDrive/gunshot_models/`

### Model Files After Training

After training on Colab, you'll find these files in your Google Drive:
- `model_best.keras`: Best model based on validation accuracy
- `model_final.keras`: Final model after training
- `logs/`: Training logs for TensorBoard

## Local Usage (After Training)

Once you have trained the model on Colab, you can use it locally:

1. Download the trained model files from Google Drive
2. Place them in the appropriate directories:
   - Place `model_best.keras` or `model_final.keras` in the `models/` directory
3. Use the model for inference using `test_detection.py`