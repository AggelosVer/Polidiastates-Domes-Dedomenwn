import pandas as pd
import numpy as np
from project1_loader import filter_movies


def save_preprocessed_dataset(input_filepath='data_movies_clean.csv', 
                               output_filepath='movies_preprocessed.csv'):
    
    print("="*70)
    print("MOVIE DATASET PREPROCESSING")
    print("="*70)
    

    print(f"\n[1/5] Loading dataset from '{input_filepath}'...")
    try:
        df = pd.read_csv(input_filepath, low_memory=False)
        print(f"       Loaded {len(df)} movies")
    except FileNotFoundError:
        print(f"       Error: File '{input_filepath}' not found")
        return None
    except Exception as e:
        print(f"       Error loading file: {e}")
        return None
    

    print("\n[2/5] Handling missing values...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    
    missing_before = df.isnull().sum().sum()
    df[numeric_cols] = df[numeric_cols].fillna(0)
    df[non_numeric_cols] = df[non_numeric_cols].fillna('Unknown')
    print(f"       Filled {missing_before} missing values")
    

    print("\n[3/5] Parsing release dates...")
    if 'release_date' in df.columns:
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        invalid_dates = df['release_date'].isnull().sum()
        df = df.dropna(subset=['release_date'])
        print(f"       Parsed dates, removed {invalid_dates} invalid entries")
        

        df['release_year'] = df['release_date'].dt.year
        print(f"       Extracted release year")
    else:
        print("       Warning: 'release_date' column not found")
    

    print("\n[4/5] Applying filtering criteria...")
    rows_before = len(df)
    df_filtered = filter_movies(df)
    
    if df_filtered is None or df_filtered.empty:
        print("       Error: No movies match the filter criteria")
        return None
    
    rows_after = len(df_filtered)
    rows_removed = rows_before - rows_after
    print(f"       Filtered dataset: {rows_after} movies retained ({rows_removed} removed)")
    

    print(f"\n[5/5] Saving preprocessed dataset to '{output_filepath}'...")
    try:
        df_filtered.to_csv(output_filepath, index=False)
        print(f"       Successfully saved {len(df_filtered)} movies")
    except Exception as e:
        print(f"       Error saving file: {e}")
        return None
    

    print("\n" + "="*70)
    print("PREPROCESSING SUMMARY")
    print("="*70)
    print(f"Input file:       {input_filepath}")
    print(f"Output file:      {output_filepath}")
    print(f"Original rows:    {len(df) + invalid_dates if 'release_date' in df.columns else len(df)}")
    print(f"Preprocessed rows: {len(df_filtered)}")
    print(f"Retention rate:   {len(df_filtered) / (len(df) + invalid_dates) * 100:.2f}%" if 'release_date' in df.columns else f"{len(df_filtered) / len(df) * 100:.2f}%")
    print(f"Columns:          {len(df_filtered.columns)}")
    print("="*70)
    
    print("\nSample of preprocessed data (first 5 rows):")
    print("-"*70)
    display_cols = ['title', 'release_year', 'popularity', 'vote_average', 'runtime', 'budget']
    display_cols = [col for col in display_cols if col in df_filtered.columns]
    print(df_filtered[display_cols].head().to_string())
    print("="*70 + "\n")
    
    return df_filtered


def main():

    INPUT_FILE = 'data_movies_clean.csv'
    OUTPUT_FILE = 'movies_preprocessed.csv'
    
    df_preprocessed = save_preprocessed_dataset(INPUT_FILE, OUTPUT_FILE)
    
    if df_preprocessed is not None:
        print(f"\n Preprocessing completed successfully!")
        print(f"  Preprocessed dataset saved to: {OUTPUT_FILE}")
    else:
        print(f"\n Preprocessing failed!")


if __name__ == "__main__":
    main()
