import pandas as pd
import re
import ast

def get_shingles(text, k=3):
    """Convert text into a set of character k-shingles."""
    if not text:
        return []
    
    # Normalize text: lower case and remove special characters
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    
    if len(text) < k:
        return [text]
    
    shingles = set()
    for i in range(len(text) - k + 1):
        shingles.add(text[i:i+k])
    return list(shingles)

def clean_and_tokenize(text, use_shingles=False, k=3):
    if pd.isna(text):
        return []
    try:
        if isinstance(text, str) and text.strip().startswith('[') and text.strip().endswith(']'):
            items = ast.literal_eval(text)
        else:
            items = [text]
    except (ValueError, SyntaxError):
        items = [text]
    
    if not isinstance(items, list):
        items = [str(items)]
        
    all_tokens = set()
    for item in items:
        if not isinstance(item, str):
            continue
            
        if use_shingles:
            shingles = get_shingles(item, k=k)
            all_tokens.update(shingles)
        else:
            item_lower = item.lower()
            item_clean = re.sub(r'[^\w\s]', '', item_lower)
            tokens = item_clean.split()
            all_tokens.update(tokens)
            
    return list(all_tokens)

def main():
    file_path = 'data_movies_clean.csv'
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {file_path}")
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return
    target_columns = ['production_company_names', 'genre_names']
    for col in target_columns:
        if col in df.columns:
            print(f"\nProcessing column: {col}")
            df[f'{col}_cleaned'] = df[col].apply(clean_and_tokenize)
            sample = df[df[col].notna()].head(5)
            for index, row in sample.iterrows():
                print(f"Original: {row[col]}")
                print(f"Cleaned:  {row[f'{col}_cleaned']}")
                print("-" * 20)
        else:
            print(f"Warning: Column {col} not found in CSV.")

if __name__ == "__main__":
    main()
