import time
import numpy as np
import json
from query_pipeline import MovieQueryPipeline

def benchmark_composite_schemes():
    csv_path = 'data_movies_clean.csv'
    numeric_attributes = ['budget', 'revenue', 'runtime', 'vote_average', 'popularity']
    
    range_min = [0.0, 0.0, 0.0, 0.0, 0.0]
    range_max = [1.0, 1.0, 1.0, 1.0, 1.0]
    
    numeric_constraints = None
    
    top_n = 10
    sample_size = 5000
    
    schemes = ['kdtree', 'quadtree', 'rangetree', 'rtree']
    results = {}
    
    for scheme in schemes:
        print(f"\n{'='*80}")
        print(f"Benchmarking {scheme.upper()} + LSH")
        print(f"{'='*80}\n")
        
        pipeline = MovieQueryPipeline(
            csv_path=csv_path,
            index_type=scheme,
            dimension=5,
            text_attribute='production_company_names',
            lsh_num_perm=128,
            lsh_threshold=0.5,
            max_entries=10,
            min_entries=4
        )
        
        load_start = time.time()
        if not pipeline.load_data(apply_filter=False, normalize=True):
            print(f"Failed to load data for {scheme}")
            continue
        
        if len(pipeline.df) > sample_size:
            print(f"Sampling {sample_size} movies from {len(pipeline.df)} total")
            pipeline.df = pipeline.df.sample(n=sample_size, random_state=42)
        
        load_time = time.time() - load_start
        
        build_start = time.time()
        if not pipeline.build_spatial_index(numeric_attributes):
            print(f"Failed to build spatial index for {scheme}")
            continue
        build_time = time.time() - build_start
        
        query_start = time.time()
        query_results = pipeline.end_to_end_query(
            numeric_constraints=numeric_constraints,
            query_type='range',
            range_min=range_min,
            range_max=range_max,
            top_n=top_n
        )
        query_time = time.time() - query_start
        
        results[scheme] = {
            'load_time': load_time,
            'build_time': build_time,
            'query_time': query_time,
            'total_time': load_time + build_time + query_time,
            'result_count': len(query_results)
        }
        
        print(f"\n{scheme.upper()} + LSH Timing Summary:")
        print(f"  Load Time:  {load_time:.4f}s")
        print(f"  Build Time: {build_time:.4f}s")
        print(f"  Query Time: {query_time:.4f}s")
        print(f"  Total Time: {results[scheme]['total_time']:.4f}s")
        print(f"  Results:    {results[scheme]['result_count']}")
    
    print(f"\n{'='*80}")
    print("COMPARATIVE RESULTS")
    print(f"{'='*80}")
    print(f"{'Scheme':<20} | {'Query Time (s)':<15} | {'Total Time (s)':<15} | {'Results':<10}")
    print("-" * 80)
    
    for scheme in schemes:
        if scheme in results:
            r = results[scheme]
            print(f"{(scheme.upper() + ' + LSH'):<20} | {r['query_time']:<15.4f} | {r['total_time']:<15.4f} | {r['result_count']:<10}")
    
    print(f"{'='*80}\n")
    
    if len(results) > 0:
        fastest_query = min(results.items(), key=lambda x: x[1]['query_time'])
        fastest_total = min(results.items(), key=lambda x: x[1]['total_time'])
        
        print(f"Fastest Query Pipeline: {fastest_query[0].upper()} + LSH ({fastest_query[1]['query_time']:.4f}s)")
        print(f"Fastest Overall:        {fastest_total[0].upper()} + LSH ({fastest_total[1]['total_time']:.4f}s)")
    
    with open('composite_benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to composite_benchmark_results.json")

if __name__ == "__main__":
    benchmark_composite_schemes()
