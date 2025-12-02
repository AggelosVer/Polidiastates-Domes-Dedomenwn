import time
import numpy as np
import pandas as pd
import ast
from typing import List, Dict, Any, Set

from project1_loader import load_and_process_data
from extract_5d_vectors import extract_5d_vectors
from preprocess_text import clean_and_tokenize
from kd_tree import KDTree
from quadtree import QuadTree
from range_tree import RangeTree
from r_tree import RTree
from lsh import MinHashLSH

def load_full_data():
    print("Loading full dataset...")
    df = load_and_process_data('data_movies_clean.csv', apply_filter=False, normalize=False)
    if df is None:
        print("Failed to load data.")
        return None, None, None
    
    if 'production_company_names' not in df.columns:
        print("Column 'production_company_names' missing.")
        return None, None, None
        
    print("Extracting vectors...")
    
    MAX_SAMPLES = 2000
    if len(df) > MAX_SAMPLES:
        print(f"Dataset too large ({len(df)}), sampling {MAX_SAMPLES} for benchmark...")
        df = df.sample(n=MAX_SAMPLES, random_state=42)
        
    vectors, reference_df, vector_df = extract_5d_vectors(df)
    
    print("\nVector Statistics (Min - Max):")
    dim_names = ['popularity', 'vote_average', 'runtime', 'budget', 'release_year']
    mins = np.min(vectors, axis=0)
    maxs = np.max(vectors, axis=0)
    for i, name in enumerate(dim_names):
        print(f"  {name}: {mins[i]:.2f} - {maxs[i]:.2f}")
    
    print("Extracting metadata and tokens...")
    metadata = {}
    text_tokens = {}
    
    def parse_countries(c_str):
        if pd.isna(c_str): return []
        if isinstance(c_str, str):
            c_str = c_str.strip("[]'\"")
            return [c.strip().strip("'\"") for c in c_str.split(',')]
        return []

    for idx, row in df.loc[vector_df.index].iterrows():
        doc_id = idx
        
        countries = parse_countries(row.get('origin_country', ''))
        lang = row.get('original_language', '')
        metadata[doc_id] = {'countries': set(countries), 'lang': lang}
        
        text_raw = row.get('production_company_names', '')
        tokens = clean_and_tokenize(text_raw)
        text_tokens[doc_id] = tokens
        
    return vectors, metadata, text_tokens, vector_df.index.tolist()

def run_benchmark():
    vectors, metadata, text_tokens, doc_ids = load_full_data()
    if vectors is None:
        return

    num_points = len(vectors)
    print(f"\nTotal Data Points: {num_points}")
    
    print("\n=== Building Indices ===")
    
    print("Building KD-Tree...")
    start = time.time()
    kd = KDTree(k=5)
    kd.build(vectors.tolist(), doc_ids)
    kd_build_time = time.time() - start
    print(f"KD-Tree built in {kd_build_time:.4f}s")
    
    print("Building QuadTree...")
    mins = np.min(vectors, axis=0)
    maxs = np.max(vectors, axis=0)
    bounds = np.column_stack((mins, maxs + 0.001))
    
    start = time.time()
    qt = QuadTree(bounds, k=5, capacity=20)
    qt.build(vectors, doc_ids)
    qt_build_time = time.time() - start
    print(f"QuadTree built in {qt_build_time:.4f}s")
    
    print("Building Range Tree...")
    start = time.time()
    rt = RangeTree(vectors.tolist(), doc_ids)
    rt_build_time = time.time() - start
    print(f"Range Tree built in {rt_build_time:.4f}s")
    
    print("Building R-Tree...")
    start = time.time()
    r_tree = RTree(max_entries=4, min_entries=2, dimension=5)
    r_tree_subset = False
    if num_points > 5000:
        print("  (Using subset of 5000 points for R-Tree due to slow insertion)")
        r_tree_indices = range(5000)
        r_tree_subset = True
    else:
        r_tree_indices = range(num_points)
        
    for i in r_tree_indices:
        r_tree.insert(vectors[i], doc_ids[i])
    r_tree_build_time = time.time() - start
    print(f"R-Tree built in {r_tree_build_time:.4f}s")
    
    print("Building LSH Index...")
    start = time.time()
    lsh = MinHashLSH(num_perm=128, threshold=0.5)
    for doc_id, tokens in text_tokens.items():
        sig = lsh.compute_signature(tokens)
        lsh.add(doc_id, sig)
    lsh_build_time = time.time() - start
    print(f"LSH built in {lsh_build_time:.4f}s")
    
    q_min = [0.6, 4.0, 60.0, -float('inf'), 1990.0]
    q_max = [20.0, 9.0, 180.0, float('inf'), 2025.0]
    
    target_countries = {'US', 'GB'}
    target_lang = 'en'
    
    import random
    query_doc_id = random.choice(doc_ids)
    query_tokens = text_tokens[query_doc_id]
    query_sig = lsh.compute_signature(query_tokens)
    print(f"\nQuery Doc ID: {query_doc_id}")
    print(f"Query Tokens: {query_tokens}")
    
    print("\n=== Executing Queries ===")
    
    results_table = []
    
    def run_scheme(name, tree_query_func, is_rtree=False):
        t0 = time.time()
        
        if is_rtree and r_tree_subset:
             pass
             
        tree_results = tree_query_func()
        t1 = time.time()
        
        filtered_ids = []
        for result in tree_results:
            if isinstance(result, tuple) and len(result) == 2:
                _, doc_id = result
            else:
                doc_id = result
            meta = metadata.get(doc_id)
            if meta:
                country_match = not meta['countries'].isdisjoint(target_countries)
                lang_match = meta['lang'] == target_lang
                
                if country_match and lang_match:
                    filtered_ids.append(doc_id)
        
        t2 = time.time()
        
        lsh_candidates = lsh.query(query_sig)
        
        final_ids = set(filtered_ids).intersection(lsh_candidates)
        
        t3 = time.time()
        
        return {
            "Scheme": name,
            "Tree_Time": t1 - t0,
            "Filter_Time": t2 - t1,
            "LSH_Time": t3 - t2,
            "Total_Time": t3 - t0,
            "Tree_Count": len(tree_results),
            "Filter_Count": len(filtered_ids),
            "Final_Count": len(final_ids)
        }

    print("Running KD-Tree + LSH...")
    res_kd = run_scheme("KD-Tree + LSH", lambda: kd.range_query(q_min, q_max))
    results_table.append(res_kd)
    
    print("Running QuadTree + LSH...")
    q_bounds = np.column_stack((q_min, q_max))
    q_bounds[3, 0] = mins[3] 
    q_bounds[3, 1] = maxs[3] 
    
    def qt_wrapper():
        pts, dat = qt.range_query(q_bounds)
        return list(zip(pts, dat))
    
    res_qt = run_scheme("QuadTree + LSH", qt_wrapper)
    results_table.append(res_qt)
    
    print("Running Range Tree + LSH...")
    res_rt = run_scheme("Range Tree + LSH", lambda: rt.query(q_min, q_max))
    results_table.append(res_rt)
    
    print("Running R-Tree + LSH...")
    res_r = run_scheme("R-Tree + LSH", lambda: r_tree.search(q_min, q_max), is_rtree=True)
    results_table.append(res_r)
    
    print("\n" + "="*80)
    print(f"{'Scheme':<20} | {'Total (s)':<10} | {'Tree (s)':<10} | {'LSH (s)':<10} | {'Final Count':<10}")
    print("-" * 80)
    for res in results_table:
        print(f"{res['Scheme']:<20} | {res['Total_Time']:<10.4f} | {res['Tree_Time']:<10.4f} | {res['LSH_Time']:<10.4f} | {res['Final_Count']:<10}")
    print("="*80)
    
    if r_tree_subset:
        print("* R-Tree results are based on a subset of data due to build time constraints.")

if __name__ == "__main__":
    run_benchmark()
