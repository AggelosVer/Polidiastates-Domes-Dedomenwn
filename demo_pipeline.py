"""
Interactive demonstration of the End-to-End Query Pipeline
Shows each step with clear output and visualization
"""

from query_pipeline import MovieQueryPipeline
import pandas as pd

def print_header(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def print_step(step_num, description):
    print(f"\n{'─'*80}")
    print(f"STEP {step_num}: {description}")
    print(f"{'─'*80}")

def print_results_table(results, max_rows=5):
    if not results:
        print("  No results to display")
        return
    
    print(f"\n  {'Rank':<6} {'Movie ID':<10} {'Similarity':<12} {'Title':<30}")
    print(f"  {'-'*6} {'-'*10} {'-'*12} {'-'*30}")
    
    for i, (movie_id, similarity, info) in enumerate(results[:max_rows], 1):
        title = info.get('title', 'N/A')
        if len(title) > 27:
            title = title[:27] + "..."
        print(f"  {i:<6} {movie_id:<10} {similarity:<12.4f} {title:<30}")

def demonstration():
    print_header("END-TO-END MOVIE QUERY PIPELINE - INTERACTIVE DEMO")
    
    print("\nThis demonstration will walk you through each step of the pipeline:")
    print("  1. Load and preprocess movie data")
    print("  2. Apply numeric constraints to filter movies")
    print("  3. Build a spatial index on 5D numeric attributes")
    print("  4. Perform a kNN query to find similar movies")
    print("  5. Apply LSH for textual similarity")
    print("  6. Return top N most similar movies")
    
    print_step(1, "Initialize Pipeline")
    
    pipeline = MovieQueryPipeline(
        csv_path='data_movies_clean.csv',
        index_type='kdtree',
        dimension=5,
        text_attribute='production_company_names',
        lsh_num_perm=64,
        lsh_threshold=0.4
    )
    
    print("  ✓ Pipeline initialized with:")
    print(f"    - Index Type: KD-Tree")
    print(f"    - Dimensions: 5D (budget, revenue, runtime, vote_average, popularity)")
    print(f"    - Text Attribute: production_company_names")
    print(f"    - LSH Permutations: 64")
    print(f"    - LSH Threshold: 0.4")
    
    print_step(2, "Load Movie Dataset")
    
    pipeline.load_data(apply_filter=False, normalize=True)
    total_movies = len(pipeline.df)
    
    print(f"  ✓ Loaded {total_movies:,} movies from dataset")
    
    print("\n  Sampling 1000 movies for demonstration...")
    pipeline.df = pipeline.df.iloc[:1000]
    
    print(f"  ✓ Working with {len(pipeline.df):,} movies")
    
    print("\n  Sample movie data:")
    sample = pipeline.df.iloc[0]
    print(f"    Title: {sample.get('title', 'N/A')}")
    print(f"    Year: {sample.get('release_year', 'N/A')}")
    print(f"    Rating: {sample.get('vote_average', 'N/A'):.4f} (normalized)")
    print(f"    Popularity: {sample.get('popularity', 'N/A'):.4f} (normalized)")
    
    print_step(3, "Build Spatial Index")
    
    numeric_attributes = ['budget', 'revenue', 'runtime', 'vote_average', 'popularity']
    
    print(f"  Building KD-Tree on attributes: {numeric_attributes}")
    
    pipeline.build_spatial_index(numeric_attributes)
    
    print(f"  ✓ KD-Tree built successfully")
    print(f"    - Total indexed points: {len(pipeline.movie_id_to_point):,}")
    
    print_step(4, "Apply Numeric Constraints")
    
    numeric_constraints = {
        'vote_average': (0.5, 0.9),
        'popularity': (0.3, 0.8)
    }
    
    print(f"  Filtering movies with constraints:")
    for attr, (min_val, max_val) in numeric_constraints.items():
        print(f"    - {attr}: [{min_val}, {max_val}]")
    
    filtered_ids = pipeline.filter_by_numeric_constraints(numeric_constraints)
    
    print(f"\n  ✓ Filtered to {len(filtered_ids):,} movies")
    print(f"    Reduction: {((1 - len(filtered_ids)/len(pipeline.df)) * 100):.1f}%")
    
    print_step(5, "Perform kNN Query")
    
    query_point = [0.3, 0.4, 0.5, 0.7, 0.6]
    k = 100
    
    print(f"  Query Point (5D): {query_point}")
    print(f"  Finding {k} nearest neighbors...")
    
    knn_results = pipeline.knn_query(query_point, k=k)
    
    print(f"\n  ✓ Found {len(knn_results):,} nearest neighbors")
    
    if knn_results:
        print(f"\n  Closest movies:")
        for i, (dist, point, movie_id) in enumerate(knn_results[:3], 1):
            title = pipeline.df.loc[movie_id].get('title', 'N/A') if movie_id in pipeline.df.index else 'N/A'
            print(f"    {i}. Distance: {dist:.4f} - {title}")
    
    print_step(6, "Compute Intersection")
    
    spatial_ids = [movie_id for _, _, movie_id in knn_results]
    intersection = list(set(filtered_ids) & set(spatial_ids))
    
    print(f"  Filtered movies: {len(filtered_ids):,}")
    print(f"  kNN results: {len(spatial_ids):,}")
    print(f"  Intersection: {len(intersection):,}")
    
    print(f"\n  ✓ {len(intersection):,} movies passed both filters")
    
    print_step(7, "Apply LSH Textual Similarity")
    
    if len(intersection) > 0:
        print(f"  Computing MinHash signatures for {len(intersection):,} movies...")
        print(f"  Text attribute: production_company_names")
        print(f"  Building LSH buckets...")
        
        lsh_results = pipeline.apply_lsh_similarity(
            movie_ids=intersection,
            top_n=5
        )
        
        print(f"\n  ✓ LSH analysis complete")
        print(f"    Found {len(lsh_results)} similar movies")
    else:
        print("  ⚠ No movies in intersection, skipping LSH")
        lsh_results = []
    
    print_step(8, "Final Results - Top Similar Movies")
    
    if lsh_results:
        final_results = []
        for movie_id, similarity, tokens in lsh_results:
            movie_info = {
                'movie_id': movie_id,
                'similarity': similarity,
                'tokens': tokens
            }
            
            if movie_id in pipeline.df.index:
                row = pipeline.df.loc[movie_id]
                movie_info['title'] = row.get('title', 'Unknown')
                movie_info['release_year'] = row.get('release_year', 'Unknown')
                movie_info['vote_average'] = row.get('vote_average', 0)
                movie_info['popularity'] = row.get('popularity', 0)
            
            final_results.append((movie_id, similarity, movie_info))
        
        print_results_table(final_results)
        
        print("\n  Detailed view of top result:")
        if final_results:
            top_movie = final_results[0][2]
            print(f"    Movie ID: {top_movie['movie_id']}")
            print(f"    Title: {top_movie.get('title', 'N/A')}")
            print(f"    Year: {top_movie.get('release_year', 'N/A')}")
            print(f"    Similarity Score: {top_movie['similarity']:.4f}")
            print(f"    Vote Average: {top_movie.get('vote_average', 0):.4f}")
            print(f"    Popularity: {top_movie.get('popularity', 0):.4f}")
            print(f"    Production Companies (tokens): {top_movie['tokens'][:5]}...")
    else:
        print("  No results found")
    
    print_header("DEMONSTRATION COMPLETE")
    
    print("\nSummary:")
    print(f"  • Started with: {len(pipeline.df):,} movies")
    print(f"  • After numeric filtering: {len(filtered_ids):,} movies")
    print(f"  • After kNN query: {len(spatial_ids):,} movies")
    print(f"  • Intersection: {len(intersection):,} movies")
    print(f"  • Final results: {len(lsh_results)} most similar movies")
    
    print("\nPipeline successfully demonstrated all 5 requirements:")
    print("  ✓ 1. Filters movies using numeric constraints")
    print("  ✓ 2. Indexes results in a chosen data structure (KD-Tree)")
    print("  ✓ 3. Performs kNN queries")
    print("  ✓ 4. Feeds the result set to LSH")
    print("  ✓ 5. Returns top N similar textual entities")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    demonstration()
