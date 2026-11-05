import pandas as pd
from sklearn.model_selection import train_test_split
import os

def prepare_data(raw_path="data/raw/properties.csv", output_dir="data/processed/"):
    print("Loading raw data...")
    df = pd.read_csv(raw_path)
    
    # Simple human-like cleaning: drop missing targets and fill basic numericals
    df = df.dropna(subset=['price'])
    df = df.fillna(0)
    
    # Feature selection
    features = ['bedrooms', 'bathrooms', 'square_feet', 'year_built']
    X = df[features]
    y = df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Save processed data for the next step in the pipeline
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)
    
    print("Data prep complete. Files saved to processed folder.")

if __name__ == "__main__":
    prepare_data()