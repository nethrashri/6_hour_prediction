import numpy as np
import tensorflow as tf
import json
import os
from collections import Counter
from datetime import datetime
import time
import sys

def predict_next_6_hours(start_timestamp, model_path="wifi_app_predictor.tflite", encoding_path="application_encoding.json"):
    """
    Predict top 5 most likely application types for the next 6 hours from a given timestamp.
    
    Args:
        start_timestamp (int): Starting timestamp in Unix time
        model_path (str): Path to the TFLite model
        encoding_path (str): Path to the application encoding JSON file
    
    Returns:
        list: Top 5 most frequently used application types for the next 6 hours
    """
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input/output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Load application encodings
    app_names = {}
    if os.path.exists(encoding_path):
        try:
            with open(encoding_path, 'r') as f:
                encodings = json.load(f)
                app_names = encodings.get('int_to_app', {})
        except Exception as e:
            print(f"Error loading application encodings: {e}")
    
    # Generate timestamps for the next 6 hours (one prediction every 30 minutes)
    interval_minutes = 30
    interval_seconds = interval_minutes * 60
    total_predictions = 12  # 6 hours with 30-minute intervals = 12 predictions
    timestamps = [start_timestamp + (i * interval_seconds) for i in range(total_predictions + 1)]
    
    # Store all predictions
    all_predictions = []
    
    # Make predictions for each timestamp
    for ts in timestamps:
        # For each timestamp, make multiple predictions with slight variations
        for j in range(3):  # 3 samples per timestamp
            # Add small random offset to timestamp (±15 minutes)
            sample_ts = ts + np.random.randint(-900, 900)
            
            # Prepare input: Shape must be (1, 1)
            input_data = np.array([[sample_ts]], dtype=np.float32)
            
            # Set input tensor
            interpreter.set_tensor(input_details[0]['index'], input_data)
            
            # Run inference
            interpreter.invoke()
            
            # Get output tensor
            output = interpreter.get_tensor(output_details[0]['index'])
            
            # Get top 3 predictions for this timestamp
            top_indices = np.argsort(output[0])[-3:][::-1]
            
            # Add predictions to our list with weighting based on confidence
            for idx in top_indices:
                confidence = output[0][idx]
                # Only include predictions with reasonable confidence
                if confidence > 0.1:  # 10% confidence threshold
                    # Weight predictions by confidence
                    weight = int(confidence * 10) + 1  # Ensure at least weight of 1
                    all_predictions.extend([int(idx)] * weight)
    
    # Count frequency of each predicted class
    prediction_counts = Counter(all_predictions)
    
    # Get the top 5 most frequent predictions (or fewer if we don't have 5)
    num_to_return = min(5, len(prediction_counts))
    top_classes = prediction_counts.most_common(num_to_return)
    
    # Format the results
    results = []
    for class_id, count in top_classes:
        app_name = app_names.get(str(class_id), f"Unknown App Type {class_id}")
        frequency = count / len(all_predictions) if all_predictions else 0
        results.append({
            "class_id": class_id,
            "app_name": app_name,
            "frequency": frequency,
            "count": count
        })
    
    return results

if __name__ == "__main__":
    # Get timestamp from command line or use current time
    if len(sys.argv) > 1:
        timestamp = int(sys.argv[1])
    else:
        timestamp = int(time.time())
    
    print(f"Predicting application usage for 6 hours starting from timestamp: {timestamp}")
    print(f"Start time: {datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time: {datetime.fromtimestamp(timestamp + 6*3600).strftime('%Y-%m-%d %H:%M:%S')}")
    
    top_apps = predict_next_6_hours(timestamp)
    
    print("\nTop 5 Predicted Applications for the next 6 hours:")
    print("-" * 60)
    for i, app in enumerate(top_apps, 1):
        print(f"{i}. {app['app_name']} (Class ID: {app['class_id']}) - " +
              f"Frequency: {app['frequency']:.2%}, Count: {app['count']}")
