"""
Comprehensive example demonstrating all features of the Movie Query Pipeline
"""

from query_pipeline import MovieQueryPipeline
import pandas as pd

def example_1_basic_range_query():
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Range Query + LSH")
    print("="*80)
    print("\nGoal: Find movies within a specific range of attributes")
    print("      and rank them by textual similarity")
    
    pipeline = MovieQueryPipeline(
        csv_path='data_movies_clean.csv',
        index_type='kdtree',
        dimension=5,
        text_attribute='production_company_names',
        lsh_num_perm=64,
        lsh_threshold=0.4
    )
    
    pipeline.load_data(apply_filter=False, normalize=True)
    pipeline.df = pipeline.df.iloc[:1000]
    
    numeric_attributes = ['budget', 'revenue', 'runtime', 'vote_average', 'popularity']
    pipeline.build_spatial_index(numeric_attributes)
    
    results = pipeline.end_to_end_query(
        query_type='range',
        range_min=[0.0, 0.0, 0.3, 0.4, 0.2],
        range_max=[0.5, 0.5, 0.8, 0.9, 0.7],
        top_n=3
    )
    
    return results

def example_2_knn_with_constraints():
    print("\n" + "="*80)
    print("EXAMPLE 2: kNN Query with Numeric Constraints")
    print("="*80)
    print("\nGoal: First filter by vote_average and popularity,")
    print("      then find nearest neighbors in 5D space,")
    print("      finally rank by textual similarity")
    
    pipeline = MovieQueryPipeline(
        csv_path='data_movies_clean.csv',
        index_type='rtree',
        dimension=5,
        text_attribute='production_company_names',
        lsh_num_perm=64,
        lsh_threshold=0.4,
        max_entries=8,
        min_entries=3
    )
    
    pipeline.load_data(apply_filter=False, normalize=True)
    pipeline.df = pipeline.df.iloc[:1000]
    
    numeric_attributes = ['budget', 'revenue', 'runtime', 'vote_average', 'popularity']
    pipeline.build_spatial_index(numeric_attributes)
    
    numeric_constraints = {
        'vote_average': (0.5, 0.9),
        'popularity': (0.3, 0.8)
    }
    
    results = pipeline.end_to_end_query(
        numeric_constraints=numeric_constraints,
        query_type='knn',
        knn_query_point=[0.3, 0.4, 0.5, 0.7, 0.6],
        knn_k=50,
        top_n=3
    )
    
    return results

def example_3_comparing_index_types():
    print("\n" + "="*80)
    print("EXAMPLE 3: Comparing Different Index Types")
    print("="*80)
    print("\nGoal: Run the same query on different index types")
    print("      to compare performance and results")
    
    index_types = ['kdtree', 'rtree', 'quadtree']
    query_point = [0.2, 0.3, 0.5, 0.6, 0.5]
    all_results = {}
    
    for idx_type in index_types:
        print(f"\n--- Testing with {idx_type.upper()} ---")
        
        if idx_type == 'quadtree':
            bounds = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
            pipeline = MovieQueryPipeline(
                csv_path='data_movies_clean.csv',
                index_type=idx_type,
                dimension=5,
                text_attribute='production_company_names',
                lsh_num_perm=64,
                lsh_threshold=0.4,
                bounds=bounds,
                capacity=15
            )
        else:
            pipeline = MovieQueryPipeline(
                csv_path='data_movies_clean.csv',
                index_type=idx_type,
                dimension=5,
                text_attribute='production_company_names',
                lsh_num_perm=64,
                lsh_threshold=0.4
            )
        
        pipeline.load_data(apply_filter=False, normalize=True)
        pipeline.df = pipeline.df.iloc[:500]
        
        numeric_attributes = ['budget', 'revenue', 'runtime', 'vote_average', 'popularity']
        pipeline.build_spatial_index(numeric_attributes)
        
        results = pipeline.end_to_end_query(
            query_type='knn',
            knn_query_point=query_point,
            knn_k=30,
            top_n=3
        )
        
        all_results[idx_type] = results
    
    return all_results

def example_4_different_text_attributes():
    print("\n" + "="*80)
    print("EXAMPLE 4: Using Different Text Attributes for LSH")
    print("="*80)
    print("\nGoal: Compare similarity based on different textual features")
    
    text_attributes = ['production_company_names', 'genres', 'original_language']
    all_results = {}
    
    for text_attr in text_attributes:
        print(f"\n--- Using text attribute: {text_attr} ---")
        
        pipeline = MovieQueryPipeline(
            csv_path='data_movies_clean.csv',
            index_type='kdtree',
            dimension=5,
            text_attribute=text_attr,
            lsh_num_perm=64,
            lsh_threshold=0.3
        )
        
        pipeline.load_data(apply_filter=False, normalize=True)
        pipeline.df = pipeline.df.iloc[:800]
        
        numeric_attributes = ['budget', 'revenue', 'runtime', 'vote_average', 'popularity']
        pipeline.build_spatial_index(numeric_attributes)
        
        try:
            results = pipeline.end_to_end_query(
                query_type='range',
                range_min=[0.1, 0.1, 0.3, 0.4, 0.3],
                range_max=[0.6, 0.6, 0.8, 0.9, 0.8],
                top_n=3
            )
            all_results[text_attr] = results
        except Exception as e:
            print(f"Error with {text_attr}: {e}")
            all_results[text_attr] = []
    
    return all_results

def example_5_progressive_filtering():
    print("\n" + "="*80)
    print("EXAMPLE 5: Progressive Filtering Pipeline")
    print("="*80)
    print("\nGoal: Demonstrate how each step reduces the result set")
    
    pipeline = MovieQueryPipeline(
        csv_path='data_movies_clean.csv',
        index_type='kdtree',
        dimension=5,
        text_attribute='production_company_names',
        lsh_num_perm=64,
        lsh_threshold=0.4
    )
    
    pipeline.load_data(apply_filter=False, normalize=True)
    pipeline.df = pipeline.df.iloc[:2000]
    
    print(f"\nStep 0: Initial dataset size: {len(pipeline.df)} movies")
    
    numeric_attributes = ['budget', 'revenue', 'runtime', 'vote_average', 'popularity']
    pipeline.build_spatial_index(numeric_attributes)
    
    numeric_constraints = {
        'vote_average': (0.6, 0.95),
        'popularity': (0.4, 0.9),
        'runtime': (0.3, 0.8)
    }
    
    filtered_ids = pipeline.filter_by_numeric_constraints(numeric_constraints)
    print(f"Step 1: After numeric filtering: {len(filtered_ids)} movies")
    
    knn_results = pipeline.knn_query([0.3, 0.4, 0.5, 0.7, 0.6], k=100)
    print(f"Step 2: After kNN query: {len(knn_results)} movies")
    
    spatial_ids = [movie_id for _, _, movie_id in knn_results]
    intersection = list(set(filtered_ids) & set(spatial_ids))
    print(f"Step 3: After intersection: {len(intersection)} movies")
    
    if len(intersection) > 0:
        lsh_results = pipeline.apply_lsh_similarity(intersection, top_n=5)
        print(f"Step 4: After LSH ranking: {len(lsh_results)} top similar movies")
    else:
        print("Step 4: No movies in intersection, skipping LSH")
        lsh_results = []
    
    return lsh_results


if __name__ == "__main__":
    print("\n" + "="*80)
    print("COMPREHENSIVE EXAMPLES: MOVIE QUERY PIPELINE")
    print("="*80)
    
    example_1_basic_range_query()
    
    example_2_knn_with_constraints()
    
    example_3_comparing_index_types()
    
    example_4_different_text_attributes()
    
    example_5_progressive_filtering()
    
    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("="*80)
