import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def filter_movies(df):
    if df is None or df.empty:
        return None
    
    filtered_df = df.copy()
    
    if 'release_year' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['release_year'] >= 2000) & 
            (filtered_df['release_year'] <= 2020)
        ]
    
    if 'popularity' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['popularity'] >= 3) & 
            (filtered_df['popularity'] <= 6)
        ]
    
    if 'vote_average' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['vote_average'] >= 3) & 
            (filtered_df['vote_average'] <= 5)
        ]
    
    if 'runtime' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['runtime'] >= 30) & 
            (filtered_df['runtime'] <= 60)
        ]
    
    if 'origin_country' in filtered_df.columns:
        def contains_country(country_str, countries):
            if pd.isna(country_str):
                return False
            if isinstance(country_str, str):
                country_str = country_str.strip("[]'\"")
                country_list = [c.strip().strip("'\"") for c in country_str.split(',')]
                return any(c in countries for c in country_list)
            elif isinstance(country_str, list):
                return any(c in countries for c in country_str)
            return False
        
        filtered_df = filtered_df[
            filtered_df['origin_country'].apply(lambda x: contains_country(x, {'US', 'GB'}))
        ]
    
    if 'original_language' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['original_language'] == 'en']
    
    return filtered_df

def load_and_process_data(filepath, apply_filter=False, normalize=True):
    try:
        df = pd.read_csv(filepath, low_memory=False)
    except FileNotFoundError:
        return None
    except Exception as e:
        return None

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns

    df[numeric_cols] = df[numeric_cols].fillna(0)
    df[non_numeric_cols] = df[non_numeric_cols].fillna('Unknown')
    
    if 'release_date' in df.columns:
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df = df.dropna(subset=['release_date'])

    if 'release_date' in df.columns:
        df['release_year'] = df['release_date'].dt.year

    if apply_filter:
        df = filter_movies(df)
        if df is None or df.empty:
            return None

    cols_to_normalize = ['budget', 'revenue', 'runtime', 'vote_average', 'vote_count', 'popularity']
    cols_to_normalize = [c for c in cols_to_normalize if c in df.columns]
    
    if cols_to_normalize and normalize:
        scaler = MinMaxScaler()
        df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
    
    return df

if __name__ == "__main__":
    FILE_PATH = 'data_movies_clean.csv' 
    processed_df = load_and_process_data(FILE_PATH, apply_filter=True)
    
    if processed_df is not None:
        print(processed_df.head())
        print(processed_df.info())
