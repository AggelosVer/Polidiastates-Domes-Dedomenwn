import pandas as pd
import numpy as np
from project1_loader import load_and_process_data


def extract_5d_vectors(df):
    required_columns = ['popularity', 'vote_average', 'runtime', 'budget', 'release_year']
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    vector_df = df[required_columns].copy()
    vector_df = vector_df.dropna()
    valid_indices = vector_df.index
    
    reference_cols = []
    if 'id' in df.columns:
        reference_cols.append('id')
    if 'title' in df.columns:
        reference_cols.append('title')
    
    if reference_cols:
        reference_df = df.loc[valid_indices, reference_cols].copy()
    else:
        reference_df = pd.DataFrame({'index': valid_indices})
    
    vectors = vector_df.to_numpy()
    
    return vectors, reference_df, vector_df


def save_vectors(vectors, reference_df, output_file='movie_5d_vectors.csv'):
    dimension_names = ['popularity', 'vote_average', 'runtime', 'budget', 'release_year']
    
    vectors_df = pd.DataFrame(vectors, columns=dimension_names, index=reference_df.index)
    result_df = pd.concat([reference_df.reset_index(drop=True), vectors_df.reset_index(drop=True)], axis=1)
    
    result_df.to_csv(output_file, index=False)
    print(f"Saved {len(vectors)} 5D vectors to {output_file}")
    
    return result_df


def print_vector_statistics(vectors, vector_df):
    dimension_names = ['popularity', 'vote_average', 'runtime', 'budget', 'release_year']
    
    print("\n" + "="*70)
    print("5D VECTOR EXTRACTION STATISTICS")
    print("="*70)
    print(f"\nTotal vectors extracted: {len(vectors)}")
    print(f"Vector shape: {vectors.shape}")
    print(f"\nVector suitable for:")
    print("  - KD-Tree indexing")
    print("  - Quadtree indexing")
    print("  - Range-Tree indexing")
    print("  - R-Tree indexing")
    
    print("\n" + "-"*70)
    print("DIMENSION STATISTICS (Normalized Values)")
    print("-"*70)
    
    for i, dim_name in enumerate(dimension_names):
        col_data = vectors[:, i]
        print(f"\n{i+1}. {dim_name.upper()}")
        print(f"   Min:    {col_data.min():.6f}")
        print(f"   Max:    {col_data.max():.6f}")
        print(f"   Mean:   {col_data.mean():.6f}")
        print(f"   Median: {np.median(col_data):.6f}")
        print(f"   Std:    {col_data.std():.6f}")
    
    print("\n" + "="*70)
    
    print("\nSAMPLE VECTORS (first 5):")
    print("-"*70)
    sample_df = vector_df.head()
    print(sample_df.to_string())
    print("="*70 + "\n")


def main():
    FILE_PATH = 'data_movies_clean.csv'
    
    print("Loading and processing movie data...")
    df = load_and_process_data(FILE_PATH, apply_filter=True)
    
    if df is None or df.empty:
        print("Error: Could not load data or no movies match the filter criteria.")
        return
    
    print(f"Loaded {len(df)} movies after filtering")
    
    print("\nExtracting 5-dimensional vectors...")
    vectors, reference_df, vector_df = extract_5d_vectors(df)
    
    print_vector_statistics(vectors, vector_df)
    
    output_file = 'movie_5d_vectors.csv'
    result_df = save_vectors(vectors, reference_df, output_file)
    
    np.save('movie_5d_vectors.npy', vectors)
    print(f"Saved vectors in binary format to movie_5d_vectors.npy")
    
    reference_df.to_csv('movie_5d_reference.csv', index=False)
    print(f"Saved reference information to movie_5d_reference.csv")
    
    return vectors, reference_df, vector_df


if __name__ == "__main__":
    vectors, reference_df, vector_df = main()
