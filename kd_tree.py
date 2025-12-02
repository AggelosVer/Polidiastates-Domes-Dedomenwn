import heapq
import math

class Node:
    def __init__(self, point, data, axis, left=None, right=None):
        self.point = point
        self.data = data
        self.axis = axis
        self.left = left
        self.right = right

class KDTree:
    def __init__(self, k=2):
        self.k = k
        self.root = None

    def build(self, points, data=None):
        if data is None:
            data = [None] * len(points)
        if len(points) != len(data):
            raise ValueError("Points and data must have same length")
        combined = list(zip(points, data))
        self.root = self._build_recursive(combined, 0)

    def _build_recursive(self, combined, depth):
        if not combined:
            return None
        axis = depth % self.k
        combined.sort(key=lambda x: x[0][axis])
        median = len(combined) // 2
        return Node(
            point=combined[median][0],
            data=combined[median][1],
            axis=axis,
            left=self._build_recursive(combined[:median], depth + 1),
            right=self._build_recursive(combined[median + 1:], depth + 1)
        )

    def insert(self, point, data=None):
        if len(point) != self.k:
            raise ValueError(f"Point must be {self.k}-dimensional")
        self.root = self._insert_recursive(self.root, point, data, 0)

    def _insert_recursive(self, node, point, data, depth):
        if node is None:
            return Node(point, data, depth % self.k)
        axis = node.axis
        if point[axis] < node.point[axis]:
            node.left = self._insert_recursive(node.left, point, data, depth + 1)
        else:
            node.right = self._insert_recursive(node.right, point, data, depth + 1)
        return node

    def delete(self, point):
        self.root = self._delete_recursive(self.root, point, 0)

    def _find_min(self, node, dim, depth):
        if node is None:
            return None
        axis = node.axis
        if axis == dim:
            if node.left is None:
                return node
            return self._find_min(node.left, dim, depth + 1)
        left_min = self._find_min(node.left, dim, depth + 1)
        right_min = self._find_min(node.right, dim, depth + 1)
        res = node
        if left_min and left_min.point[dim] < res.point[dim]:
            res = left_min
        if right_min and right_min.point[dim] < res.point[dim]:
            res = right_min
        return res

    def _delete_recursive(self, node, point, depth):
        if node is None:
            return None
        axis = node.axis
        if node.point == point:
            if node.right is not None:
                min_node = self._find_min(node.right, axis, depth + 1)
                node.point = min_node.point
                node.right = self._delete_recursive(node.right, min_node.point, depth + 1)
            elif node.left is not None:
                min_node = self._find_min(node.left, axis, depth + 1)
                node.point = min_node.point
                node.right = self._delete_recursive(node.left, min_node.point, depth + 1)
                node.left = None
            else:
                return None
            return node
        if point[axis] < node.point[axis]:
            node.left = self._delete_recursive(node.left, point, depth + 1)
        else:
            node.right = self._delete_recursive(node.right, point, depth + 1)
        return node

    def update(self, old_point, new_point):
        self.delete(old_point)
        self.insert(new_point)

    def range_query(self, range_min, range_max):
        results = []
        self._range_query_recursive(self.root, range_min, range_max, results)
        return results

    def _range_query_recursive(self, node, range_min, range_max, results):
        if node is None:
            return
        in_range = True
        for i in range(self.k):
            if not (range_min[i] <= node.point[i] <= range_max[i]):
                in_range = False
                break
        if in_range:
            results.append((node.point, node.data))
        axis = node.axis
        if range_min[axis] <= node.point[axis]:
            self._range_query_recursive(node.left, range_min, range_max, results)
        if range_max[axis] >= node.point[axis]:
            self._range_query_recursive(node.right, range_min, range_max, results)

    def knn_query(self, target, k):
        heap = []
        self._knn_recursive(self.root, target, k, heap)
        return sorted([(-h[0], h[1]) for h in heap], key=lambda x: x[0])

    def _knn_recursive(self, node, target, k, heap):
        if node is None:
            return
        dist_sq = sum((node.point[i] - target[i])**2 for i in range(self.k))
        dist = math.sqrt(dist_sq)
        if len(heap) < k:
            heapq.heappush(heap, (-dist, node.point))
        else:
            if dist < -heap[0][0]:
                heapq.heapreplace(heap, (-dist, node.point))
        axis = node.axis
        diff = target[axis] - node.point[axis]
        close_node = node.left if diff < 0 else node.right
        far_node = node.right if diff < 0 else node.left
        self._knn_recursive(close_node, target, k, heap)
        if len(heap) < k or abs(diff) < -heap[0][0]:
            self._knn_recursive(far_node, target, k, heap)

if __name__ == "__main__":
    points = [(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)]
    data = ["A", "B", "C", "D", "E", "F"]
    tree = KDTree(k=2)
    tree.build(points, data)
    print("Initial points:", points)
    print("Range Query [0,0] to [6,6]:", tree.range_query((0, 0), (6, 6)))
    print("Insert (3, 3)")
    tree.insert((3, 3), "G")
    print("Range Query [0,0] to [6,6]:", tree.range_query((0, 0), (6, 6)))
    print("Delete (5, 4)")
    tree.delete((5, 4))
    print("Range Query [0,0] to [6,6]:", tree.range_query((0, 0), (6, 6)))
    print("KNN Query target (9, 2) k=3:", tree.knn_query((9, 2), 3))
    print("Update (2, 3) -> (2, 4)")
    tree.update((2, 3), (2, 4))
    print("Range Query [0,0] to [6,6]:", tree.range_query((0, 0), (6, 6)))
