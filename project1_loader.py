import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_process_data(filepath):
    try:
        df = pd.read_csv(filepath, low_memory=False)
        print(f"Successfully loaded data from {filepath}. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns

    df[numeric_cols] = df[numeric_cols].fillna(0)
    df[non_numeric_cols] = df[non_numeric_cols].fillna('Unknown')
    
    if 'release_date' in df.columns:
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        initial_rows = len(df)
        df = df.dropna(subset=['release_date'])
        dropped_rows = initial_rows - len(df)
        if dropped_rows > 0:
            print(f"Dropped {dropped_rows} rows due to missing or invalid release_date.")
    else:
        print("Warning: 'release_date' column not found.")

    if 'release_date' in df.columns:
        df['release_year'] = df['release_date'].dt.year
        print("Created 'release_year' column.")

    cols_to_normalize = ['budget', 'revenue', 'runtime', 'vote_average', 'vote_count', 'popularity']
    cols_to_normalize = [c for c in cols_to_normalize if c in df.columns]
    
    if cols_to_normalize:
        scaler = MinMaxScaler()
        df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
        print(f"Normalized columns: {cols_to_normalize}")
    
    return df

if __name__ == "__main__":
    FILE_PATH = 'data_movies_clean.csv' 
    processed_df = load_and_process_data(FILE_PATH)
    
    if processed_df is not None:
        print("\nFirst 5 rows of processed data:")
        print(processed_df.head())
        print("\nData Info:")
        print(processed_df.info())
