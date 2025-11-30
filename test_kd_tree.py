import pytest
import math
from kd_tree import KDTree

@pytest.fixture
def sample_points():
    return [(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)]

@pytest.fixture
def tree(sample_points):
    t = KDTree(k=2)
    t.build(list(sample_points))
    return t

def test_build_and_range_query(tree):
    results = tree.range_query((0, 0), (10, 10))
    assert len(results) == 6
    expected = set([(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)])
    assert set(results) == expected

def test_range_query_specific(tree):
    results = tree.range_query((1, 2), (6, 5))
    expected = set([(2, 3), (5, 4)])
    assert set(results) == expected

def test_insert(tree):
    tree.insert((3, 3))
    results = tree.range_query((0, 0), (10, 10))
    assert len(results) == 7
    assert (3, 3) in results
    
    results_small = tree.range_query((2.5, 2.5), (3.5, 3.5))
    assert (3, 3) in results_small

def test_delete_leaf(tree):
    tree.delete((9, 6))
    results = tree.range_query((0, 0), (10, 10))
    assert len(results) == 5
    assert (9, 6) not in results

def test_delete_root_or_internal(tree):
    to_delete = (7, 2)
    tree.delete(to_delete)
    
    results = tree.range_query((0, 0), (10, 10))
    assert len(results) == 5
    assert to_delete not in results
    
    assert (2, 3) in results
    assert (8, 1) in results

def test_delete_not_existing(tree):
    tree.delete((100, 100))
    results = tree.range_query((0, 0), (10, 10))
    assert len(results) == 6

def test_update(tree):
    old_p = (2, 3)
    new_p = (2, 5)
    tree.update(old_p, new_p)
    
    results = tree.range_query((0, 0), (10, 10))
    assert len(results) == 6
    assert old_p not in results
    assert new_p in results

def test_knn_exact_match(tree):
    target = (5, 4)
    neighbors = tree.knn_query(target, k=1)
    assert len(neighbors) == 1
    dist, point = neighbors[0]
    assert point == target
    assert dist == 0.0

def test_knn_general(tree):
    target = (9, 2)
    neighbors = tree.knn_query(target, k=3)
    
    assert len(neighbors) == 3
    
    d1, p1 = neighbors[0]
    assert p1 == (8, 1)
    assert math.isclose(d1, math.sqrt(2))
    
    d2, p2 = neighbors[1]
    assert p2 == (7, 2)
    assert math.isclose(d2, 2.0)
    
    d3, p3 = neighbors[2]
    assert p3 == (9, 6)
    assert math.isclose(d3, 4.0)

def test_knn_k_larger_than_size(tree):
    target = (0, 0)
    neighbors = tree.knn_query(target, k=10)
    assert len(neighbors) == 6

def test_high_dimensions():
    points = [
        (1, 2, 3),
        (4, 5, 6),
        (7, 8, 9),
        (1, 1, 1)
    ]
    t = KDTree(k=3)
    t.build(points)
    
    res = t.range_query((0, 0, 0), (5, 5, 5))
    assert len(res) == 2
    assert (1, 2, 3) in res
    assert (1, 1, 1) in res
    
    neighbors = t.knn_query((0, 0, 0), k=2)
    assert neighbors[0][1] == (1, 1, 1)
    assert neighbors[1][1] == (1, 2, 3)
