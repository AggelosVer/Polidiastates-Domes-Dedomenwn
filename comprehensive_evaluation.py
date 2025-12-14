"""
Comprehensive Evaluation Script for Multi-dimensional Indexing + LSH

Tests all 4 schemes (KD-Tree, QuadTree, Range Tree, R-Tree) combined with LSH
using properly filtered data according to project specifications.
"""

import time
import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from collections import defaultdict

from project1_loader import load_and_process_data, filter_movies
from extract_5d_vectors import extract_5d_vectors
from preprocess_text import clean_and_tokenize
from kd_tree import KDTree
from quadtree import QuadTree
from range_tree import RangeTree
from r_tree import RTree
from lsh import MinHashLSH
from json_formatter import format_query_results_to_json, batch_format_queries_to_json


def load_filtered_data():
    """Load and filter data according to project specifications."""
    print("=" * 80)
    print("LOADING AND FILTERING DATA")
    print("=" * 80)
    
    print("\n1. Loading full dataset...")
    df = load_and_process_data('data_movies_clean.csv', apply_filter=False, normalize=False)
    if df is None:
        print("❌ Failed to load data.")
        return None, None, None, None, None
    
    print(f"   Loaded: {len(df)} total movies")
    
    # Apply project-specified filters
    print("\n2. Applying project filters:")
    print("   - Release year: 2000-2020")
    print("   - Popularity: 3-6")
    print("   - Vote average: 3-5")
    print("   - Runtime: 30-60 minutes")
    print("   - Origin country: US or GB")
    print("   - Original language: en")
    
    df_filtered = filter_movies(df)
    
    if df_filtered is None or len(df_filtered) == 0:
        print("❌ No movies match the filter criteria.")
        return None, None, None, None, None
    
    print(f"   Filtered: {len(df_filtered)} movies match criteria")
    
    # Further filter: only movies with production_company_names
    print("\n3. Filtering for non-empty production_company_names...")
    df_filtered = df_filtered[
        df_filtered['production_company_names'].notna() & 
        (df_filtered['production_company_names'] != '') &
        (df_filtered['production_company_names'] != 'Unknown')
    ]
    print(f"   With production data: {len(df_filtered)} movies")
    
    if len(df_filtered) == 0:
        print("❌ No movies with production company data.")
        return None, None, None, None, None
    
    # Sample if too large
    MAX_SAMPLES = 3000
    if len(df_filtered) > MAX_SAMPLES:
        print(f"\n4. Sampling {MAX_SAMPLES} movies for evaluation...")
        df_filtered = df_filtered.sample(n=MAX_SAMPLES, random_state=42)
    
    print(f"\n✅ Final dataset: {len(df_filtered)} movies")
    
    # Extract 5D vectors
    print("\n5. Extracting 5D vectors...")
    vectors, reference_df, vector_df = extract_5d_vectors(df_filtered)
    
    print("\n   Vector Statistics:")
    dim_names = ['popularity', 'vote_average', 'runtime', 'budget', 'release_year']
    for i, name in enumerate(dim_names):
        min_val = np.min(vectors[:, i])
        max_val = np.max(vectors[:, i])
        print(f"   - {name:15s}: [{min_val:8.2f}, {max_val:8.2f}]")
    
    # Extract text tokens
    print("\n6. Extracting production company tokens...")
    text_tokens = {}
    valid_doc_ids = []
    
    for idx in vector_df.index:
        if idx not in df_filtered.index:
            continue
        
        row = df_filtered.loc[idx]
        text_raw = row.get('production_company_names', '')
        tokens = clean_and_tokenize(text_raw)
        
        if tokens:  # Only include if tokens exist
            text_tokens[idx] = tokens
            valid_doc_ids.append(idx)
    
    print(f"   Movies with valid tokens: {len(valid_doc_ids)}")
    
    # Filter vectors to only valid documents
    valid_indices = [i for i, doc_id in enumerate(vector_df.index) if doc_id in valid_doc_ids]
    vectors = vectors[valid_indices]
    
    return vectors, text_tokens, valid_doc_ids, df_filtered


def build_indices(vectors, doc_ids, text_tokens):
    """Build all 4 index structures and LSH."""
    print("\n" + "=" * 80)
    print("BUILDING INDICES")
    print("=" * 80)
    
    build_times = {}
    
    # KD-Tree
    print("\n1. Building KD-Tree...")
    start = time.time()
    kd = KDTree(k=5)
    kd.build(vectors.tolist(), doc_ids)
    build_times['kd_tree'] = time.time() - start
    print(f"   ✅ Built in {build_times['kd_tree']:.4f}s")
    
    # QuadTree
    print("\n2. Building QuadTree...")
    mins = np.min(vectors, axis=0)
    maxs = np.max(vectors, axis=0)
    bounds = np.column_stack((mins, maxs + 0.001))
    
    start = time.time()
    qt = QuadTree(bounds, k=5, capacity=20)
    qt.build(vectors, doc_ids)
    build_times['quadtree'] = time.time() - start
    print(f"   ✅ Built in {build_times['quadtree']:.4f}s")
    
    # Range Tree
    print("\n3. Building Range Tree...")
    start = time.time()
    rt = RangeTree(vectors.tolist(), doc_ids)
    build_times['range_tree'] = time.time() - start
    print(f"   ✅ Built in {build_times['range_tree']:.4f}s")
    
    # R-Tree
    print("\n4. Building R-Tree...")
    start = time.time()
    r_tree = RTree(max_entries=4, min_entries=2, dimension=5)
    
    # Limit R-Tree to 3000 points if dataset is larger
    num_points = len(vectors)
    if num_points > 3000:
        print(f"   (Using first 3000 points due to slow insertion)")
        r_tree_indices = range(3000)
    else:
        r_tree_indices = range(num_points)
    
    for i in r_tree_indices:
        r_tree.insert(vectors[i], doc_ids[i])
    
    build_times['r_tree'] = time.time() - start
    print(f"   ✅ Built in {build_times['r_tree']:.4f}s")
    
    # LSH
    print("\n5. Building LSH Index...")
    start = time.time()
    lsh = MinHashLSH(num_perm=128, threshold=0.5)
    for doc_id, tokens in text_tokens.items():
        sig = lsh.compute_signature(tokens)
        lsh.add(doc_id, sig)
    build_times['lsh'] = time.time() - start
    print(f"   ✅ Built in {build_times['lsh']:.4f}s")
    
    indices = {
        'kd_tree': kd,
        'quadtree': (qt, bounds, mins, maxs),
        'range_tree': rt,
        'r_tree': r_tree,
        'lsh': lsh
    }
    
    return indices, build_times


def select_query_movies(doc_ids, text_tokens, df, num_queries=5):
    """Select diverse query movies with good token coverage."""
    print("\n" + "=" * 80)
    print("SELECTING QUERY MOVIES")
    print("=" * 80)
    
    # Sort by number of tokens (descending) to get movies with rich data
    doc_token_counts = [(doc_id, len(text_tokens[doc_id])) for doc_id in doc_ids]
    doc_token_counts.sort(key=lambda x: x[1], reverse=True)
    
    # Select top N with different token counts for diversity
    query_movies = []
    step = max(1, len(doc_token_counts) // (num_queries * 2))
    
    for i in range(0, min(len(doc_token_counts), num_queries * step), step):
        doc_id, token_count = doc_token_counts[i]
        if len(query_movies) >= num_queries:
            break
        query_movies.append(doc_id)
    
    print(f"\n✅ Selected {len(query_movies)} query movies:\n")
    for i, doc_id in enumerate(query_movies, 1):
        title = df.loc[doc_id, 'title'] if doc_id in df.index and 'title' in df.columns else 'Unknown'
        tokens = text_tokens[doc_id]
        print(f"{i}. ID {doc_id:6d} | {title[:50]:50s} | {len(tokens):2d} tokens")
    
    return query_movies


def run_query(scheme_name, query_func, lsh, query_sig, query_tokens):
    """Run a single query and return results with timing."""
    start = time.time()
    
    # Phase 1: Tree query
    tree_results = query_func()
    tree_time = time.time() - start
    
    # Extract doc_ids from results
    tree_doc_ids = []
    for result in tree_results:
        if isinstance(result, tuple) and len(result) == 2:
            _, doc_id = result
        else:
            doc_id = result
        tree_doc_ids.append(doc_id)
    
    # Phase 2: LSH similarity filtering
    lsh_start = time.time()
    lsh_candidates = lsh.query(query_sig)
    
    # Find intersection
    final_doc_ids = set(tree_doc_ids).intersection(lsh_candidates)
    
    # Compute similarities for final candidates
    similarities = []
    for doc_id in final_doc_ids:
        if doc_id in lsh.signatures:
            sim = lsh.get_jaccard_sim(query_sig, lsh.signatures[doc_id])
            similarities.append((doc_id, sim))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    lsh_time = time.time() - lsh_start
    total_time = time.time() - start
    
    return {
        'scheme': scheme_name,
        'tree_count': len(tree_doc_ids),
        'lsh_candidates': len(lsh_candidates),
        'final_count': len(similarities),
        'tree_time': tree_time,
        'lsh_time': lsh_time,
        'total_time': total_time,
        'results': similarities  # List of (doc_id, similarity)
    }


def run_all_queries(indices, query_movies, text_tokens, df, vectors, doc_ids, top_n=10):
    """Run all queries for all schemes."""
    print("\n" + "=" * 80)
    print("RUNNING QUERIES")
    print("=" * 80)
    
    kd = indices['kd_tree']
    qt, bounds, mins, maxs = indices['quadtree']
    rt = indices['range_tree']
    r_tree = indices['r_tree']
    lsh = indices['lsh']
    
    # Define range query bounds (from project specs)
    q_min = [3.0, 3.0, 30.0, -float('inf'), 2000.0]
    q_max = [6.0, 5.0, 60.0, float('inf'), 2020.0]
    
    all_results = defaultdict(dict)  # {query_id: {scheme_name: results}}
    
    for query_idx, query_doc_id in enumerate(query_movies, 1):
        print(f"\n{'-' * 80}")
        print(f"Query {query_idx}/{len(query_movies)}: Movie ID {query_doc_id}")
        
        title = df.loc[query_doc_id, 'title'] if query_doc_id in df.index else 'Unknown'
        print(f"Title: {title}")
        
        query_tokens = text_tokens[query_doc_id]
        query_sig = lsh.compute_signature(query_tokens)
        print(f"Tokens ({len(query_tokens)}): {query_tokens[:10]}{'...' if len(query_tokens) > 10 else ''}")
        
        # KD-Tree
        print("\n  → KD-Tree + LSH...", end=" ")
        res_kd = run_query(
            "KD-Tree + LSH",
            lambda: kd.range_query(q_min, q_max),
            lsh, query_sig, query_tokens
        )
        print(f"✓ ({res_kd['final_count']} results in {res_kd['total_time']:.4f}s)")
        all_results[query_doc_id]['kd_tree'] = res_kd
        
        # QuadTree
        print("  → QuadTree + LSH...", end=" ")
        q_bounds = np.column_stack((q_min, q_max))
        q_bounds[3, 0] = mins[3]
        q_bounds[3, 1] = maxs[3]
        
        def qt_query_wrapper():
            pts, dat = qt.range_query(q_bounds)
            if len(pts) > 0:
                return list(zip(pts, dat))
            return []
        
        res_qt = run_query(
            "QuadTree + LSH",
            qt_query_wrapper,
            lsh, query_sig, query_tokens
        )
        print(f"✓ ({res_qt['final_count']} results in {res_qt['total_time']:.4f}s)")
        all_results[query_doc_id]['quadtree'] = res_qt
        
        # Range Tree
        print("  → Range Tree + LSH...", end=" ")
        res_rt = run_query(
            "Range Tree + LSH",
            lambda: rt.query(q_min, q_max),
            lsh, query_sig, query_tokens
        )
        print(f"✓ ({res_rt['final_count']} results in {res_rt['total_time']:.4f}s)")
        all_results[query_doc_id]['range_tree'] = res_rt
        
        # R-Tree
        print("  → R-Tree + LSH...", end=" ")
        res_r = run_query(
            "R-Tree + LSH",
            lambda: r_tree.search(q_min, q_max),
            lsh, query_sig, query_tokens
        )
        print(f"✓ ({res_r['final_count']} results in {res_r['total_time']:.4f}s)")
        all_results[query_doc_id]['r_tree'] = res_r
    
    return all_results


def save_results_as_json(all_results, text_tokens, df, output_dir='evaluation_results'):
    """Save all results as JSON files."""
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for query_id, schemes in all_results.items():
        for scheme_name, result in schemes.items():
            # Convert results to expected format for json_formatter
            top_results = []
            for doc_id, similarity in result['results'][:10]:  # Top 10
                tokens = text_tokens.get(doc_id, [])
                top_results.append((doc_id, similarity, tokens))
            
            # Generate JSON
            filename = f"{output_dir}/query_{query_id}_{scheme_name}.json"
            json_str = format_query_results_to_json(
                top_results,
                df=df,
                query_movie_id=query_id,
                include_titles=True,
                include_tokens=False,
                output_file=filename,
                pretty_print=True
            )
    
    print(f"\n✅ Saved individual query results to {output_dir}/")
    
    # Create performance summary
    summary = {
        "queries": [],
        "scheme_comparison": {}
    }
    
    for query_id, schemes in all_results.items():
        query_summary = {
            "query_id": int(query_id),
            "title": str(df.loc[query_id, 'title']) if query_id in df.index else "Unknown",
            "schemes": {}
        }
        
        for scheme_name, result in schemes.items():
            query_summary["schemes"][scheme_name] = {
                "total_time": round(result['total_time'], 4),
                "tree_time": round(result['tree_time'], 4),
                "lsh_time": round(result['lsh_time'], 4),
                "tree_count": result['tree_count'],
                "final_count": result['final_count']
            }
        
        summary["queries"].append(query_summary)
    
    # Save summary
    summary_file = f"{output_dir}/performance_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved performance summary to {summary_file}")


def print_performance_summary(all_results, build_times):
    """Print performance comparison table."""
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    
    print("\n1. Build Times:")
    print(f"   KD-Tree:     {build_times['kd_tree']:8.4f}s")
    print(f"   QuadTree:    {build_times['quadtree']:8.4f}s")
    print(f"   Range Tree:  {build_times['range_tree']:8.4f}s")
    print(f"   R-Tree:      {build_times['r_tree']:8.4f}s")
    print(f"   LSH:         {build_times['lsh']:8.4f}s")
    
    print("\n2. Average Query Performance:")
    
    scheme_names = ['kd_tree', 'quadtree', 'range_tree', 'r_tree']
    scheme_display = ['KD-Tree + LSH', 'QuadTree + LSH', 'Range Tree + LSH', 'R-Tree + LSH']
    
    for scheme_name, display_name in zip(scheme_names, scheme_display):
        times = [results[scheme_name]['total_time'] for results in all_results.values()]
        counts = [results[scheme_name]['final_count'] for results in all_results.values()]
        
        avg_time = np.mean(times)
        avg_count = np.mean(counts)
        
        print(f"   {display_name:20s}: {avg_time:8.4f}s avg | {avg_count:6.1f} results avg")
    
    print("\n" + "=" * 80)


def main():
    """Main evaluation workflow."""
    print("\n" + "=" * 80)
    print("  COMPREHENSIVE EVALUATION: Multi-dimensional Indexing + LSH")
    print("=" * 80)
    
    # 1. Load and filter data
    data = load_filtered_data()
    if data[0] is None:
        print("\n❌ Evaluation aborted: No valid data.")
        return
    
    vectors, text_tokens, doc_ids, df = data
    
    # 2. Build indices
    indices, build_times = build_indices(vectors, doc_ids, text_tokens)
    
    # 3. Select query movies
    query_movies = select_query_movies(doc_ids, text_tokens, df, num_queries=5)
    
    # 4. Run all queries
    all_results = run_all_queries(indices, query_movies, text_tokens, df, vectors, doc_ids, top_n=10)
    
    # 5. Save results
    save_results_as_json(all_results, text_tokens, df)
    
    # 6. Print summary
    print_performance_summary(all_results, build_times)
    
    print("\n✅ EVALUATION COMPLETE!")
    print("   Check the 'evaluation_results/' directory for detailed JSON outputs.\n")


if __name__ == "__main__":
    main()
