import pandas as pd
import numpy as np
from query_pipeline import MovieQueryPipeline

print("=" * 80)
print("QUICK TEST: MOVIE QUERY PIPELINE")
print("=" * 80)

pipeline = MovieQueryPipeline(
    csv_path='data_movies_clean.csv',
    index_type='kdtree',
    dimension=5,
    text_attribute='production_company_names',
    lsh_num_perm=64,
    lsh_threshold=0.4
)

if not pipeline.load_data(apply_filter=False, normalize=True):
    print("Failed to load data")
    exit(1)

print(f"\nSampling 500 movies for quick demonstration...")
pipeline.df = pipeline.df.iloc[:500]

numeric_attributes = ['budget', 'revenue', 'runtime', 'vote_average', 'popularity']

if not pipeline.build_spatial_index(numeric_attributes):
    print("Failed to build spatial index")
    exit(1)

print("\n" + "=" * 80)
print("TEST: Range Query + LSH")
print("=" * 80)

results = pipeline.end_to_end_query(
    query_type='range',
    range_min=[0.0, 0.0, 0.0, 0.3, 0.2],
    range_max=[0.4, 0.4, 0.6, 0.8, 0.7],
    top_n=5
)

print(f"\nSuccessfully retrieved {len(results)} results!")

print("\n" + "=" * 80)
print("TEST: kNN Query + LSH")
print("=" * 80)

knn_query_point = [0.2, 0.3, 0.5, 0.6, 0.5]

results2 = pipeline.end_to_end_query(
    query_type='knn',
    knn_query_point=knn_query_point,
    knn_k=50,
    top_n=5
)

print(f"\nSuccessfully retrieved {len(results2)} results!")

print("\n" + "=" * 80)
print("QUICK TEST COMPLETE - PIPELINE WORKING!")
print("=" * 80)
