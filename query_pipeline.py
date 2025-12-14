import pandas as pd
import numpy as np
from typing import List, Tuple, Any, Optional, Dict
from index_manager import IndexManager
from lsh import MinHashLSH, find_top_n_similar_movies
from project1_loader import load_and_process_data
import ast
import re

class MovieQueryPipeline:
    def __init__(self, 
                 csv_path: str,
                 index_type: str = 'rtree',
                 dimension: int = 5,
                 text_attribute: str = 'production_company_names',
                 lsh_num_perm: int = 128,
                 lsh_threshold: float = 0.5,
                 **index_kwargs):
        self.csv_path = csv_path
        self.index_type = index_type
        self.dimension = dimension
        self.text_attribute = text_attribute
        self.lsh_num_perm = lsh_num_perm
        self.lsh_threshold = lsh_threshold
        self.index_kwargs = index_kwargs
        
        self.df = None
        self.index_manager = None
        self.lsh = None
        self.movie_id_to_point = {}
        self.point_to_movie_id = {}
        
        print(f"Initializing MovieQueryPipeline with {index_type} index")
        
    def load_data(self, apply_filter: bool = False, normalize: bool = True) -> bool:
        print(f"\nLoading movie data from {self.csv_path}...")
        self.df = load_and_process_data(self.csv_path, apply_filter=apply_filter, normalize=normalize)
        
        if self.df is None or self.df.empty:
            print("Failed to load data or no data available after filtering")
            return False
        
        print(f"Loaded {len(self.df)} movies")
        return True
    
    def build_spatial_index(self, numeric_attributes: List[str]) -> bool:
        if self.df is None:
            print("Error: Data not loaded. Call load_data() first.")
            return False
        
        if len(numeric_attributes) != self.dimension:
            print(f"Error: Expected {self.dimension} attributes, got {len(numeric_attributes)}")
            return False
        
        for attr in numeric_attributes:
            if attr not in self.df.columns:
                print(f"Error: Attribute '{attr}' not found in dataset")
                return False
        
        print(f"\nBuilding {self.index_type} index with attributes: {numeric_attributes}")
        
        self.index_manager = IndexManager(
            index_type=self.index_type,
            dimension=self.dimension,
            **self.index_kwargs
        )
        
        points = []
        movie_ids = []
        
        for idx, row in self.df.iterrows():
            point = [float(row[attr]) for attr in numeric_attributes]
            points.append(point)
            movie_ids.append(idx)
            self.movie_id_to_point[idx] = point
            self.point_to_movie_id[tuple(point)] = idx
        
        self.index_manager.build(points, movie_ids)
        
        print(f"Built index with {len(points)} points")
        print(f"Index info: {self.index_manager.get_info()}")
        
        return True
    
    def filter_by_numeric_constraints(self, constraints: Dict[str, Tuple[float, float]]) -> List[int]:
        if self.df is None:
            print("Error: Data not loaded. Call load_data() first.")
            return []
        
        print(f"\nApplying numeric constraints: {constraints}")
        
        filtered_df = self.df.copy()
        
        for attr, (min_val, max_val) in constraints.items():
            if attr not in filtered_df.columns:
                print(f"Warning: Attribute '{attr}' not found, skipping constraint")
                continue
            
            filtered_df = filtered_df[
                (filtered_df[attr] >= min_val) & (filtered_df[attr] <= max_val)
            ]
        
        movie_ids = filtered_df.index.tolist()
        print(f"Filtered to {len(movie_ids)} movies")
        
        return movie_ids
    
    def range_query(self, range_min: List[float], range_max: List[float]) -> List[Tuple[List[float], int]]:
        if self.index_manager is None:
            print("Error: Spatial index not built. Call build_spatial_index() first.")
            return []
        
        print(f"\nPerforming range query: [{range_min}, {range_max}]")
        
        results = self.index_manager.range_query(range_min, range_max)
        
        print(f"Range query returned {len(results)} results")
        
        return results
    
    def knn_query(self, query_point: List[float], k: int = 10) -> List[Tuple[float, List[float], int]]:
        if self.index_manager is None:
            print("Error: Spatial index not built. Call build_spatial_index() first.")
            return []
        
        print(f"\nPerforming kNN query: point={query_point}, k={k}")
        
        results = self.index_manager.knn_query(query_point, k)
        
        print(f"kNN query returned {len(results)} results")
        
        return results
    
    def apply_lsh_similarity(self, 
                           movie_ids: List[int],
                           query_movie_id: Optional[int] = None,
                           top_n: int = 10) -> List[Tuple[int, float, List[str]]]:
        if self.df is None:
            print("Error: Data not loaded. Call load_data() first.")
            return []
        
        if not movie_ids:
            print("Error: No movie IDs provided for LSH")
            return []
        
        print(f"\nApplying LSH on {len(movie_ids)} movies using '{self.text_attribute}'")
        
        if query_movie_id is None:
            query_movie_id = movie_ids[0]
            print(f"Using first movie ID as query: {query_movie_id}")
        
        try:
            results = find_top_n_similar_movies(
                filtered_movie_ids=movie_ids,
                df=self.df,
                text_attribute=self.text_attribute,
                query_movie_id=query_movie_id,
                top_n=top_n,
                num_perm=self.lsh_num_perm,
                threshold=self.lsh_threshold
            )
            
            return results
        except Exception as e:
            print(f"Error during LSH: {e}")
            return []
    
    def end_to_end_query(self,
                        numeric_constraints: Optional[Dict[str, Tuple[float, float]]] = None,
                        query_type: str = 'range',
                        range_min: Optional[List[float]] = None,
                        range_max: Optional[List[float]] = None,
                        knn_query_point: Optional[List[float]] = None,
                        knn_k: int = 50,
                        query_movie_id: Optional[int] = None,
                        top_n: int = 10) -> List[Tuple[int, float, Dict[str, Any]]]:
        print("\n" + "=" * 80)
        print("END-TO-END QUERY PIPELINE")
        print("=" * 80)
        
        movie_ids = []
        
        if numeric_constraints:
            movie_ids = self.filter_by_numeric_constraints(numeric_constraints)
            if not movie_ids:
                print("No movies match the numeric constraints")
                return []
        else:
            movie_ids = self.df.index.tolist()
        
        spatial_results = []
        
        if query_type == 'range':
            if range_min is None or range_max is None:
                print("Error: range_min and range_max required for range query")
                return []
            
            spatial_results = self.range_query(range_min, range_max)
            
        elif query_type == 'knn':
            if knn_query_point is None:
                print("Error: knn_query_point required for kNN query")
                return []
            
            spatial_results = self.knn_query(knn_query_point, knn_k)
        
        else:
            print(f"Error: Unknown query_type '{query_type}'. Use 'range' or 'knn'")
            return []
        
        if not spatial_results:
            print("Spatial query returned no results")
            return []
        
        if query_type == 'range':
            spatial_movie_ids = [movie_id for _, movie_id in spatial_results]
        else:
            spatial_movie_ids = [movie_id for _, _, movie_id in spatial_results]
        
        indexed_movie_ids = list(set(movie_ids) & set(spatial_movie_ids))
        
        print(f"\nIntersection of filtered movies and spatial results: {len(indexed_movie_ids)} movies")
        
        if not indexed_movie_ids:
            print("No movies in the intersection set")
            return []
        
        lsh_results = self.apply_lsh_similarity(
            movie_ids=indexed_movie_ids,
            query_movie_id=query_movie_id,
            top_n=top_n
        )
        
        final_results = []
        for movie_id, similarity, tokens in lsh_results:
            movie_info = {
                'movie_id': movie_id,
                'similarity': similarity,
                'tokens': tokens
            }
            
            if movie_id in self.df.index:
                row = self.df.loc[movie_id]
                movie_info['title'] = row.get('title', 'Unknown')
                movie_info['release_year'] = row.get('release_year', 'Unknown')
                movie_info['vote_average'] = row.get('vote_average', 'Unknown')
                movie_info['popularity'] = row.get('popularity', 'Unknown')
                movie_info['point'] = self.movie_id_to_point.get(movie_id, [])
            
            final_results.append((movie_id, similarity, movie_info))
        
        print("\n" + "=" * 80)
        print(f"FINAL RESULTS: TOP {len(final_results)} SIMILAR MOVIES")
        print("=" * 80)
        
        for rank, (movie_id, similarity, info) in enumerate(final_results, 1):
            print(f"\n{rank}. Movie ID: {movie_id}")
            print(f"   Title: {info.get('title', 'N/A')}")
            print(f"   Year: {info.get('release_year', 'N/A')}")
            print(f"   Similarity: {similarity:.4f}")
            print(f"   Rating: {info.get('vote_average', 'N/A')}")
            print(f"   Popularity: {info.get('popularity', 'N/A')}")
            print(f"   Tokens: {info.get('tokens', [])[:5]}...")
        
        print("\n" + "=" * 80)
        
        return final_results


if __name__ == "__main__":
    print("=" * 80)
    print("MOVIE QUERY PIPELINE DEMONSTRATION")
    print("=" * 80)
    
    pipeline = MovieQueryPipeline(
        csv_path='data_movies_clean.csv',
        index_type='rtree',
        dimension=5,
        text_attribute='production_company_names',
        lsh_num_perm=128,
        lsh_threshold=0.5,
        max_entries=10,
        min_entries=4
    )
    
    if not pipeline.load_data(apply_filter=False, normalize=True):
        print("Failed to load data")
        exit(1)
    
    numeric_attributes = ['budget', 'revenue', 'runtime', 'vote_average', 'popularity']
    
    if not pipeline.build_spatial_index(numeric_attributes):
        print("Failed to build spatial index")
        exit(1)
    
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Range Query + LSH")
    print("=" * 80)
    
    numeric_constraints = {
        'vote_average': (0.4, 0.8),
        'popularity': (0.3, 0.7)
    }
    
    results1 = pipeline.end_to_end_query(
        numeric_constraints=numeric_constraints,
        query_type='range',
        range_min=[0.0, 0.0, 0.0, 0.4, 0.3],
        range_max=[0.5, 0.5, 1.0, 0.8, 0.7],
        top_n=5
    )
    
    print("\n" + "=" * 80)
    print("EXAMPLE 2: kNN Query + LSH")
    print("=" * 80)
    
    knn_query_point = [0.2, 0.3, 0.5, 0.6, 0.5]
    
    results2 = pipeline.end_to_end_query(
        query_type='knn',
        knn_query_point=knn_query_point,
        knn_k=100,
        top_n=5
    )
    
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Combined Constraints + kNN + LSH")
    print("=" * 80)
    
    numeric_constraints_2 = {
        'runtime': (0.2, 0.8),
        'vote_average': (0.5, 0.9)
    }
    
    results3 = pipeline.end_to_end_query(
        numeric_constraints=numeric_constraints_2,
        query_type='knn',
        knn_query_point=[0.3, 0.4, 0.5, 0.7, 0.6],
        knn_k=50,
        top_n=5
    )
    
    print("\n" + "=" * 80)
    print("PIPELINE DEMONSTRATION COMPLETE")
    print("=" * 80)
