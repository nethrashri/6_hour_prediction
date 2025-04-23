import pandas as pd
import numpy as np
from datetime import datetime
import json

def preprocess_wifi_data(input_file="custom_balanced_wifi_usage_dataset.csv", output_file="wifi_dataset.csv"):
    """Simple preprocessing script to convert timestamps to epoch time and encode application types"""
    
    print("Loading dataset...")
    df = pd.read_csv(input_file)
    print(f"Raw dataset shape: {df.shape}")
    
    # Convert timestamps to epoch time
    def convert_to_epoch(timestamp_str):
        try:
            # Try common formats
            formats = [
                '%Y-%m-%d %H:%M:%S',
                '%Y/%m/%d %H:%M:%S',
                '%d-%m-%Y %H:%M:%S',
                '%m/%d/%Y %H:%M:%S'
            ]
            
            for fmt in formats:
                try:
                    dt = datetime.strptime(timestamp_str, fmt)
                    return int(dt.timestamp())
                except ValueError:
                    continue
            
            # Try direct timestamp
            return int(float(timestamp_str))
            
        except Exception as e:
            print(f"Error converting timestamp '{timestamp_str}': {e}")
            return 0
    
    # Convert timestamps
    print("Converting timestamps...")
    df['TimestampInt'] = df['Timestamp'].apply(convert_to_epoch)
    
    # Encode application types
    print("Encoding application types...")
    app_types = df['Application_Type'].unique()
    app_to_int = {app: i for i, app in enumerate(app_types)}
    int_to_app = {str(i): app for app, i in app_to_int.items()}
    
    df['ApplicationTypeEncoded'] = df['Application_Type'].map(app_to_int)
    
    # Save encodings
    encodings = {
        'app_to_int': app_to_int,
        'int_to_app': int_to_app
    }
    
    with open('application_encoding.json', 'w') as f:
        json.dump(encodings, f, indent=2)
    
    print(f"Saved application encodings with {len(app_types)} unique types.")
    
    # Create model dataset
    model_df = df[['TimestampInt', 'ApplicationTypeEncoded']].copy()
    model_df = model_df.dropna()
    
    # Save preprocessed data
    model_df.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to {output_file}")
    
    return model_df

if __name__ == "__main__":
    preprocess_wifi_data()
