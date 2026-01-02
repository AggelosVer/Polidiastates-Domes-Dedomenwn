import time
import numpy as np
import json
from query_pipeline import MovieQueryPipeline

def diagnostic_benchmark():
    csv_path = 'data_movies_clean.csv'
    numeric_attributes = ['budget', 'revenue', 'runtime', 'vote_average', 'popularity']
    sample_size = 5000
    
    print("Running diagnostic benchmark...\n")
    
    pipeline = MovieQueryPipeline(
        csv_path=csv_path,
        index_type='kdtree',
        dimension=5,
        text_attribute='production_company_names',
        lsh_num_perm=128,
        lsh_threshold=0.5
    )
    
    load_start = time.time()
    if not pipeline.load_data(apply_filter=False, normalize=True):
        print("Failed to load data")
        return
    
    if len(pipeline.df) > sample_size:
        print(f"Sampling {sample_size} movies from {len(pipeline.df)} total")
        pipeline.df = pipeline.df.sample(n=sample_size, random_state=42)
    
    load_time = time.time() - load_start
    print(f"Load time: {load_time:.4f}s\n")
    
    build_start = time.time()
    if not pipeline.build_spatial_index(numeric_attributes):
        print("Failed to build spatial index")
        return
    build_time = time.time() - build_start
    print(f"Build time: {build_time:.4f}s\n")
    
    print("Testing range query without constraints...")
    query_start = time.time()
    range_results = pipeline.range_query([0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0])
    query_time = time.time() - query_start
    print(f"Range query returned {len(range_results)} results in {query_time:.4f}s\n")
    
    print("Testing with just constraints (no range query)...")
    constraints = {
        'vote_average': (0.2, 0.9),
        'popularity': (0.1, 0.9)
    }
    filtered_ids = pipeline.filter_by_numeric_constraints(constraints)
    print(f"Numeric constraints returned {len(filtered_ids)} movies\n")
    
    if len(range_results) > 0:
        range_movie_ids = [movie_id for _, movie_id in range_results]
        intersection = set(filtered_ids) & set(range_movie_ids)
        print(f"Intersection of constraints and range: {len(intersection)} movies\n")
        
        if len(intersection) > 0:
            print("Testing LSH on intersection...")
            lsh_results = pipeline.apply_lsh_similarity(
                movie_ids=list(intersection),
                query_movie_id=None,
                top_n=10
            )
            print(f"LSH returned {len(lsh_results)} results\n")

if __name__ == "__main__":
    diagnostic_benchmark()
