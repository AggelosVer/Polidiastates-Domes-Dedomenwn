import pandas as pd
from project1_loader import load_and_process_data

df = load_and_process_data('data_movies_clean.csv', apply_filter=False, normalize=True)

print(f"Total movies: {len(df)}")
print(f"\nDataFrame columns: {df.columns.tolist()}")

if 'vote_average' in df.columns:
    print(f"\nvote_average stats:")
    print(df['vote_average'].describe())
    print(f"Min: {df['vote_average'].min()}, Max: {df['vote_average'].max()}")
    
    va_filtered = df[(df['vote_average'] >= 0.2) & (df['vote_average'] <= 0.9)]
    print(f"\nMovies with vote_average in [0.2, 0.9]: {len(va_filtered)}")

if 'popularity' in df.columns:
    print(f"\npopularity stats:")
    print(df['popularity'].describe())
    print(f"Min: {df['popularity'].min()}, Max: {df['popularity'].max()}")
    
    pop_filtered = df[(df['popularity'] >= 0.1) & (df['popularity'] <= 0.9)]
    print(f"\nMovies with popularity in [0.1, 0.9]: {len(pop_filtered)}")

if 'vote_average' in df.columns and 'popularity' in df.columns:
    both_filtered = df[
        (df['vote_average'] >= 0.2) & (df['vote_average'] <= 0.9) &
        (df['popularity'] >= 0.1) & (df['popularity'] <= 0.9)
    ]
    print(f"\nMovies matching both constraints: {len(both_filtered)}")

sample = df.sample(n=5000, random_state=42)
print(f"\nAfter sampling 5000:")
if 'vote_average' in sample.columns and 'popularity' in sample.columns:
    both_filtered_sample = sample[
        (sample['vote_average'] >= 0.2) & (sample['vote_average'] <= 0.9) &
        (sample['popularity'] >= 0.1) & (sample['popularity'] <= 0.9)
    ]
    print(f"Movies matching both constraints in sample: {len(both_filtered_sample)}")
