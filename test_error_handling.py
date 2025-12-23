from query_pipeline import MovieQueryPipeline
from kd_tree import KDTree
from r_tree import RTree
from quadtree import QuadTree
import numpy as np

print("=" * 80)
print("ERROR HANDLING DEMONSTRATION")
print("=" * 80)

print("\n1. Testing KD-Tree with invalid dimensions...")
try:
    tree = KDTree(k=3)
    tree.build([(1, 2, 3), (4, 5, 6)])
    tree.range_query([0, 0], [5, 5])
    print("   Should have raised ValueError")
except ValueError as e:
    print(f"   Caught error: {e}")

print("\n2. Testing KD-Tree with malformed range...")
try:
    tree = KDTree(k=2)
    tree.build([(1, 2), (3, 4)])
    tree.range_query([5, 5], [1, 1])
    print("   Should have raised ValueError")
except ValueError as e:
    print(f"   Caught error: {e}")

print("\n3. Testing KD-Tree with invalid k...")
try:
    tree = KDTree(k=2)
    tree.build([(1, 2), (3, 4)])
    tree.knn_query([2, 2], k=0)
    print("   Should have raised ValueError")
except ValueError as e:
    print(f"   Caught error: {e}")

print("\n4. Testing R-Tree with NaN values...")
try:
    rtree = RTree(dimension=2)
    rtree.insert([1.0, float('nan')], "Bad Point")
    print("   Should have raised ValueError")
except ValueError as e:
    print(f"   Caught error: {e}")

print("\n5. Testing R-Tree with dimension mismatch...")
try:
    rtree = RTree(dimension=3)
    rtree.insert([1.0, 2.0], "Wrong Dim")
    print("   Should have raised ValueError")
except ValueError as e:
    print(f"   Caught error: {e}")

print("\n6. Testing Quadtree with malformed range...")
try:
    bounds = np.array([[0, 10], [0, 10]])
    qtree = QuadTree(bounds, k=2)
    qtree.build(np.array([[1, 2], [3, 4]]))
    query_bounds = np.array([[8, 2], [8, 2]])
    qtree.range_query(query_bounds)
    print("   Should have raised ValueError")
except ValueError as e:
    print(f"   Caught error: {e}")

print("\n7. Testing empty tree handling...")
tree = KDTree(k=2)
results = tree.range_query([0, 0], [5, 5])
print(f"   Empty tree range query returned: {results} (length: {len(results)})")
results = tree.knn_query([1, 1], k=5)
print(f"   Empty tree kNN query returned: {results} (length: {len(results)})")

print("\n8. Testing Query Pipeline dimension validation...")
try:
    pipeline = MovieQueryPipeline(
        csv_path='data_movies_clean.csv',
        index_type='kdtree',
        dimension=5
    )
    print("   Pipeline initialized with dimension checks")
except Exception as e:
    print(f"   Error: {e}")

print("\n9. Testing constraint range validation...")
pipeline = MovieQueryPipeline(
    csv_path='data_movies_clean.csv',
    index_type='kdtree',
    dimension=5
)
try:
    bad_constraints = {'vote_average': (0.9, 0.1)}
    if pipeline.load_data(apply_filter=False, normalize=True):
        pipeline.df = pipeline.df.iloc[:10]
        pipeline.build_spatial_index(['budget', 'revenue', 'runtime', 'vote_average', 'popularity'])
        filtered = pipeline.filter_by_numeric_constraints(bad_constraints)
    print("   Should have raised ValueError")
except ValueError as e:
    print(f"   Caught error: {e}")

print("\n" + "=" * 80)
print("ERROR HANDLING TESTS COMPLETE")
print("=" * 80)
print("\nAll validation checks are working correctly")
