import pytest
from kd_tree import KDTree
import math

class TestKDTreeInsert:
    def test_insert_2d_single_point(self):
        tree = KDTree(k=2)
        tree.build([(5, 5)], ["A"])
        tree.insert((3, 3), "B")
        results = tree.range_query((0, 0), (10, 10))
        assert len(results) == 2
        points = [r[0] for r in results]
        assert (3, 3) in points
        assert (5, 5) in points

    def test_insert_3d_multiple_points(self):
        tree = KDTree(k=3)
        tree.build([(1, 2, 3)], ["A"])
        tree.insert((4, 5, 6), "B")
        tree.insert((7, 8, 9), "C")
        results = tree.range_query((0, 0, 0), (10, 10, 10))
        assert len(results) == 3

    def test_insert_5d_point(self):
        tree = KDTree(k=5)
        tree.build([(1, 2, 3, 4, 5)], ["A"])
        tree.insert((2, 3, 4, 5, 6), "B")
        results = tree.range_query((0, 0, 0, 0, 0), (10, 10, 10, 10, 10))
        assert len(results) == 2

    def test_insert_with_data(self):
        tree = KDTree(k=2)
        tree.build([(1, 1)], ["point1"])
        tree.insert((2, 2), "point2")
        results = tree.range_query((0, 0), (5, 5))
        data_values = [r[1] for r in results]
        assert "point1" in data_values
        assert "point2" in data_values

class TestKDTreeDelete:
    def test_delete_single_point(self):
        tree = KDTree(k=2)
        tree.build([(2, 3), (5, 4), (9, 6)], ["A", "B", "C"])
        tree.delete((5, 4))
        results = tree.range_query((0, 0), (10, 10))
        assert len(results) == 2
        points = [r[0] for r in results]
        assert (5, 4) not in points

    def test_delete_nonexistent_point(self):
        tree = KDTree(k=2)
        tree.build([(1, 1), (2, 2)], ["A", "B"])
        tree.delete((10, 10))
        results = tree.range_query((0, 0), (5, 5))
        assert len(results) == 2

    def test_delete_root(self):
        tree = KDTree(k=2)
        tree.build([(5, 5)], ["A"])
        tree.delete((5, 5))
        results = tree.range_query((0, 0), (10, 10))
        assert len(results) == 0

    def test_delete_3d_point(self):
        tree = KDTree(k=3)
        tree.build([(1, 2, 3), (4, 5, 6), (7, 8, 9)], ["A", "B", "C"])
        tree.delete((4, 5, 6))
        results = tree.range_query((0, 0, 0), (10, 10, 10))
        assert len(results) == 2

    def test_delete_multiple_points(self):
        tree = KDTree(k=2)
        tree.build([(1, 1), (2, 2), (3, 3), (4, 4)], ["A", "B", "C", "D"])
        tree.delete((2, 2))
        tree.delete((4, 4))
        results = tree.range_query((0, 0), (5, 5))
        assert len(results) == 2

class TestKDTreeUpdate:
    def test_update_2d_point(self):
        tree = KDTree(k=2)
        tree.build([(2, 3), (5, 4)], ["A", "B"])
        tree.update((2, 3), (3, 4))
        results = tree.range_query((0, 0), (10, 10))
        points = [r[0] for r in results]
        assert (2, 3) not in points
        assert (3, 4) in points
        assert len(results) == 2

    def test_update_with_data(self):
        tree = KDTree(k=2)
        tree.build([(1, 1)], ["old_data"])
        tree.insert((1, 1), "new_data")
        results = tree.range_query((0, 0), (2, 2))
        assert len(results) == 2

    def test_update_3d_point(self):
        tree = KDTree(k=3)
        tree.build([(1, 2, 3), (4, 5, 6)], ["A", "B"])
        tree.update((1, 2, 3), (7, 8, 9))
        results = tree.range_query((0, 0, 0), (10, 10, 10))
        points = [r[0] for r in results]
        assert (1, 2, 3) not in points
        assert (7, 8, 9) in points

    def test_update_5d_point(self):
        tree = KDTree(k=5)
        tree.build([(1, 2, 3, 4, 5)], ["A"])
        tree.update((1, 2, 3, 4, 5), (2, 3, 4, 5, 6))
        results = tree.range_query((0, 0, 0, 0, 0), (10, 10, 10, 10, 10))
        points = [r[0] for r in results]
        assert (1, 2, 3, 4, 5) not in points
        assert (2, 3, 4, 5, 6) in points

class TestKDTreeRangeQuery:
    def test_range_query_2d(self):
        tree = KDTree(k=2)
        tree.build([(2, 3), (5, 4), (9, 6), (4, 7), (8, 1)], ["A", "B", "C", "D", "E"])
        results = tree.range_query((0, 0), (6, 6))
        assert len(results) >= 2

    def test_range_query_empty(self):
        tree = KDTree(k=2)
        tree.build([(2, 3), (5, 4)], ["A", "B"])
        results = tree.range_query((10, 10), (20, 20))
        assert len(results) == 0

    def test_range_query_all_points(self):
        tree = KDTree(k=2)
        tree.build([(1, 1), (2, 2), (3, 3)], ["A", "B", "C"])
        results = tree.range_query((0, 0), (10, 10))
        assert len(results) == 3

    def test_range_query_3d(self):
        tree = KDTree(k=3)
        tree.build([(1, 2, 3), (4, 5, 6), (7, 8, 9)], ["A", "B", "C"])
        results = tree.range_query((0, 0, 0), (5, 5, 5))
        assert len(results) >= 1

    def test_range_query_exact_boundary(self):
        tree = KDTree(k=2)
        tree.build([(2, 2), (5, 5)], ["A", "B"])
        results = tree.range_query((2, 2), (5, 5))
        assert len(results) == 2

class TestKDTreeKNN:
    def test_knn_2d_single_neighbor(self):
        tree = KDTree(k=2)
        tree.build([(2, 3), (5, 4), (9, 6)], ["A", "B", "C"])
        results = tree.knn_query((3, 3), 1)
        assert len(results) == 1
        dist, point = results[0]
        assert point == (2, 3)

    def test_knn_2d_three_neighbors(self):
        tree = KDTree(k=2)
        tree.build([(2, 3), (5, 4), (9, 6), (4, 7), (8, 1)], ["A", "B", "C", "D", "E"])
        results = tree.knn_query((4, 4), 3)
        assert len(results) == 3
        distances = [r[0] for r in results]
        assert distances == sorted(distances)

    def test_knn_exact_point(self):
        tree = KDTree(k=2)
        tree.build([(5, 5), (10, 10)], ["A", "B"])
        results = tree.knn_query((5, 5), 1)
        dist, point = results[0]
        assert dist == 0.0
        assert point == (5, 5)

    def test_knn_3d(self):
        tree = KDTree(k=3)
        tree.build([(1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4)], ["A", "B", "C", "D"])
        results = tree.knn_query((2.5, 2.5, 2.5), 2)
        assert len(results) == 2
        points = [r[1] for r in results]
        assert (2, 2, 2) in points
        assert (3, 3, 3) in points

    def test_knn_5d(self):
        tree = KDTree(k=5)
        points_5d = [(1, 2, 3, 4, 5), (2, 3, 4, 5, 6), (3, 4, 5, 6, 7)]
        tree.build(points_5d, ["A", "B", "C"])
        results = tree.knn_query((2, 3, 4, 5, 6), 1)
        assert len(results) == 1
        dist, point = results[0]
        assert dist == 0.0

    def test_knn_distance_ordering(self):
        tree = KDTree(k=2)
        tree.build([(0, 0), (1, 1), (2, 2), (3, 3)], ["A", "B", "C", "D"])
        results = tree.knn_query((0, 0), 4)
        distances = [r[0] for r in results]
        for i in range(len(distances) - 1):
            assert distances[i] <= distances[i + 1]

class TestKDTreeBuild:
    def test_build_2d(self):
        tree = KDTree(k=2)
        points = [(2, 3), (5, 4), (9, 6)]
        tree.build(points, ["A", "B", "C"])
        results = tree.range_query((0, 0), (10, 10))
        assert len(results) == 3

    def test_build_with_data(self):
        tree = KDTree(k=2)
        points = [(1, 1), (2, 2)]
        data = ["point_1", "point_2"]
        tree.build(points, data)
        results = tree.range_query((0, 0), (5, 5))
        result_data = [r[1] for r in results]
        assert "point_1" in result_data
        assert "point_2" in result_data

    def test_build_5d(self):
        tree = KDTree(k=5)
        points = [(1, 2, 3, 4, 5), (6, 7, 8, 9, 10)]
        tree.build(points)
        results = tree.range_query((0, 0, 0, 0, 0), (20, 20, 20, 20, 20))
        assert len(results) == 2

    def test_build_empty(self):
        tree = KDTree(k=2)
        tree.build([], [])
        results = tree.range_query((0, 0), (10, 10))
        assert len(results) == 0

class TestKDTreeEdgeCases:
    def test_dimension_validation(self):
        tree = KDTree(k=2)
        tree.build([(1, 1)], ["A"])
        with pytest.raises(ValueError):
            tree.insert((1, 1, 1), "B")

    def test_knn_k_larger_than_points(self):
        tree = KDTree(k=2)
        tree.build([(1, 1), (2, 2)], ["A", "B"])
        results = tree.knn_query((0, 0), 10)
        assert len(results) == 2

    def test_duplicate_points(self):
        tree = KDTree(k=2)
        tree.build([(1, 1), (1, 1)], ["A", "B"])
        results = tree.range_query((0, 0), (2, 2))
        assert len(results) == 2

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
