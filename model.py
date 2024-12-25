import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50V2

class GunShotDetector:
    def __init__(self, input_shape=(224, 224, 3)):
        """Initialize the gunshot detector model"""
        self.input_shape = input_shape
        self.model = self._build_model()
        
    def _build_model(self):
        """Build the model architecture"""
        # Use ResNet50V2 as base model
        base_model = ResNet50V2(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape
        )
        
        # Freeze the base model
        base_model.trainable = False
        
        # Create model
        model = models.Sequential([
            # Base model
            base_model,
            
            # Add custom layers
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            
            # First dense layer with strong regularization
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Second dense layer
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Output layer
            layers.Dense(2, activation='softmax')  # Changed to 2 classes with softmax
        ])
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, train_data, train_labels, validation_data=None, epochs=50, batch_size=32, callbacks=None):
        """Train the model"""
        return self.model.fit(
            train_data,
            train_labels,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
    
    def evaluate(self, test_data, test_labels):
        """Evaluate the model"""
        return self.model.evaluate(test_data, test_labels)
    
    def predict(self, data):
        """Make predictions"""
        return self.model.predict(data)
    
    def save_model(self, filepath):
        """Save the model"""
        self.model.save(filepath)
    
    @staticmethod
    def load_model(filepath):
        """Load a saved model"""
        return tf.keras.models.load_model(filepath)

if __name__ == "__main__":
    # Test model creation
    model = GunShotDetector()
    model.model.summary() 