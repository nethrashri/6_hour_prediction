import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import json

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# Define TFLite-compatible transformer-like block
class TFLiteTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, ff_dim, **kwargs):
        super().__init__(**kwargs)
        
        # Feature transformation layers
        self.dense1 = tf.keras.layers.Dense(embed_dim, activation='relu')
        self.dense2 = tf.keras.layers.Dense(ff_dim, activation='relu')
        self.dense3 = tf.keras.layers.Dense(embed_dim)
        
        # Normalization
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, inputs):
        # Feature transformation
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        
        # Residual connection
        return self.norm(x + inputs)

def train_wifi_model(data_path="wifi_dataset.csv"):
    """Train a TFLite model to predict application type from timestamp"""
    
    print("Loading dataset...")
    df = pd.read_csv(data_path)
    
    # Prepare data
    X = df['TimestampInt'].values.reshape(-1, 1).astype(np.float32)
    y = df['ApplicationTypeEncoded'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Get number of classes
    num_classes = len(np.unique(y))
    print(f"Training model for {num_classes} application types...")
    
    # Build model
    embed_dim = 64
    ff_dim = 128
    
    # Input layer
    inputs = tf.keras.layers.Input(shape=(1,), dtype=tf.float32)
    
    # Embedding layer
    x = tf.keras.layers.Dense(embed_dim)(inputs)
    
    # Transformer-like block
    x = TFLiteTransformerBlock(embed_dim=embed_dim, ff_dim=ff_dim)(x)
    
    # Classification head
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    print("Training model...")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=15,
        batch_size=32,
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Save model in TF format
    model.save('wifi_model.keras')
    
    # Convert to TFLite
    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.allow_custom_ops = False
    
    tflite_model = converter.convert()
    
    # Save the TFLite model
    with open('wifi_app_predictor.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print("TFLite model saved successfully!")
    
    # Test the model
    interpreter = tf.lite.Interpreter(model_path="wifi_app_predictor.tflite")
    interpreter.allocate_tensors()
    
    # Get test input
    test_input = np.array([[X_test[0][0]]], dtype=np.float32)
    
    # Run inference
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    
    output = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output, axis=1)[0]
    
    # Load encodings
    try:
        with open('application_encoding.json', 'r') as f:
            encodings = json.load(f)
            app_name = encodings['int_to_app'][str(int(predicted_class))]
            print(f"Test prediction: Class {predicted_class} ({app_name})")
    except:
        print(f"Test prediction: Class {predicted_class}")
    
    print("Training complete!")
    return model

if __name__ == "__main__":
    train_wifi_model()
