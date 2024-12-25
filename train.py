import os
import h5py
import numpy as np
from pathlib import Path
import tensorflow as tf
from model import GunShotDetector
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

def load_dataset(dataset_path):
    """Load the preprocessed dataset"""
    with h5py.File(dataset_path, 'r') as f:
        train_features = np.array(f['train_features'])
        train_labels = np.array(f['train_labels'])
        test_features = np.array(f['test_features'])
        test_labels = np.array(f['test_labels'])
    
    # Convert labels to categorical
    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=2)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=2)
    
    return (train_features, train_labels), (test_features, test_labels)

def create_callbacks(checkpoint_dir):
    """Create training callbacks"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    callbacks = [
        # Model checkpoint to save best models
        ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'model_best.keras'),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate when training plateaus
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        # TensorBoard logging
        TensorBoard(
            log_dir=os.path.join(checkpoint_dir, 'logs'),
            histogram_freq=1
        )
    ]
    return callbacks

def train_model():
    """Main training function"""
    try:
        print("Loading dataset...")
        dataset_path = "dataset/processed/gunshot_detection_dataset.h5"
        (train_features, train_labels), (test_features, test_labels) = load_dataset(dataset_path)
        
        print("\nDataset statistics:")
        print(f"Training samples: {len(train_features)}")
        print(f"Test samples: {len(test_features)}")
        print(f"Feature shape: {train_features.shape[1:]}")
        
        print("\nInitializing model...")
        model = GunShotDetector(num_classes=2)
        
        # Create callbacks
        checkpoint_dir = "checkpoints"
        callbacks = create_callbacks(checkpoint_dir)
        
        print("\nStarting training...")
        history = model.train(
            train_features,
            train_labels,
            validation_data=(test_features, test_labels),
            epochs=50,
            batch_size=32,
            callbacks=callbacks
        )
        
        # Evaluate final model
        print("\nEvaluating model...")
        test_loss, test_accuracy = model.model.evaluate(test_features, test_labels)
        print(f"\nFinal test accuracy: {test_accuracy*100:.2f}%")
        print(f"Final test loss: {test_loss:.4f}")
        
        # Save final model
        final_model_path = os.path.join(checkpoint_dir, "model_final.keras")
        model.save_model(final_model_path)
        print(f"\nFinal model saved to: {final_model_path}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user!")
        print("Latest best model is saved in the checkpoints directory.")
    except Exception as e:
        print(f"\nTraining stopped due to error: {str(e)}")
        print("Latest best model is saved in the checkpoints directory.")
    finally:
        print("\nTraining session ended. You can resume from the last checkpoint if needed.")

if __name__ == "__main__":
    # Enable memory growth for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    train_model() 