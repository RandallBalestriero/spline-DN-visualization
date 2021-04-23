import numpy as np
from collections import deque
from . import layers


class Point(object):
    __array_priority__ = 1000

    def __init__(self, *coordinates):
        if len(coordinates) == 1:
            if type(coordinates) == np.array:
                self.coordinates = coordinates[0]
                return
        self.coordinates = np.array(coordinates)

    def __eq__(self, other):
        return np.allclose(self.coordinates, other.coordinates)

    def __len__(self):
        return len(self.coordinates)

    def __add__(self, other):
        if hasattr(other, "coordinates"):
            return Point(*(self.coordinates + other.coordinates))
        else:
            return Point(*(self.coordinates + other))

    def __sub__(self, other):
        if hasattr(other, "coordinates"):
            return Point(*(self.coordinates - other.coordinates))
        else:
            return Point(*(self.coordinates - other))

    def __mul__(self, scalar):
        assert np.isscalar(scalar)
        return Point(*(self.coordinates * scalar))

    def __rmul__(self, scalar):
        assert np.isscalar(scalar)
        return Point(*(self.coordinates * scalar))

    def dot(self, other):
        if hasattr(other, "coordinates"):
            return np.dot(self.coordinates, other.coordinates)
        else:
            return np.dot(self.coordinates, other)

    def __getitem__(self, indices):
        return self.coordinates[indices]

    def __str__(self):
        return "Point:" + str(self.coordinates)

    def __repr__(self):
        return self.__str__()

    def norm(self, ord=2):
        return np.linalg.norm(self.coordinates, ord=ord)

    def angle(self, other=None, radian=False, signed=False):
        if other is not None:
            rad = other.angle(radian=True, signed=True) - self.angle(
                radian=True, signed=True
            )
        elif len(self.coordinates) == 2:
            rad = np.angle(self.coordinates[0] + 1j * self.coordinates[1])
        if not signed and rad < 0:
            rad += 2 * np.pi
        if not radian:
            rad = (rad / np.pi) * 180
        return rad


class Segment:
    def __init__(self, pointa, pointb):
        if not isinstance(pointa, Point):
            pointa = Point(*pointa)
        if not isinstance(pointb, Point):
            pointb = Point(*pointb)

        self.start = pointa
        self.end = pointb

    def __contains__(self, point, margin=1e-7):
        if isinstance(point, Point):
            alpha = self.closest_alpha(point)
            best_point = self.start * (1 - alpha) + self.end * alpha
            return (best_point - point).norm() <= margin
        else:
            raise RuntimeError

    def intersect(self, segment):
        assert isinstance(segment, Segment)
        da = self.end - self.start
        db = segment.end - segment.start
        dp = self.start - segment.start
        dap = Point(-da[1], da[0])
        denom = dap.dot(db)
        num = dap.dot(dp)
        if np.abs(denom) < 1e-5:
            return None
        point = (num / denom.astype(float)) * db + segment.start
        if point in self and point in segment:
            return point
        else:
            return None

    def closest_alpha(self, point):
        if point == self.start:
            return 0
        elif point == self.end:
            return 1

        diff = self.end - self.start
        if diff.norm() < 1e-7:
            return None
        alpha = (point - self.start).dot(diff) / diff.norm() ** 2
        return np.clip(alpha, 0, 1)

    def __str__(self):
        return "{}->{}".format(self.start, self.end)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, Segment):
            raise RuntimeError
        return self.start == other.start and self.end == other.end


class OrderedPath:
    def __init__(self, segments=None, vertices=None):
        if vertices is not None:
            self.segments = []
            for start, end in zip(vertices[:-1], vertices[1:]):
                if (
                    type(start) == np.ndarray
                    and np.linalg.norm(start - end) < 1e-7
                ):
                    continue
                elif isinstance(start, Point) and (start - end).norm() < 1e-7:
                    continue
                self.segments.append(Segment(start, end))

        else:
            self.segments = list(segments)
        # member variable to keep track of current index
        self._index = 0

    def __iter__(self):
        return iter(self.segments)

    def __next__(self):
        if self._index < len(self.segments):
            return self.segments[self._index]
        self._index = 0
        raise StopIteration

    def __contains__(self, point):
        if isinstance(point, Point):
            for seg in self.segments:
                if point in seg:
                    return True
            return False
        elif isinstance(point, Segment):
            for seg in self.segments:
                if point == seg:
                    return True
            return False
        elif isinstance(point, OrderedPath):
            for s in point.segments:
                if s not in self:
                    return False
            return True
        else:
            print(point)
            raise RuntimeError

    def __len__(self):
        return len(self.segments) + 1

    @property
    def start(self):
        return self.segments[0].start

    @property
    def closed(self):
        return self.segments[0].start == self.segments[-1].end

    @property
    def end(self):
        return self.segments[-1].end

    @property
    def vertices(self):
        # this could be optimized by computing it once
        # and then updating it if nodes are added
        nodes = []
        for s in self.segments:
            nodes.append(s.start)
        nodes.append(self.segments[-1].end)
        return nodes

    def intersect(self, other):
        points = []
        if isinstance(other, Segment):
            for seg in self.segments:
                point = seg.intersect(other)
                if point is not None:
                    points.append(point)

        elif isinstance(other, OrderedPath):
            for seg in other:
                points_ = self.intersect(seg)
                if points_ is not None:
                    points += points_

        if len(points):
            return points
        return None

    def insert_point(self, point):
        if point in self.vertices:
            return
        for i in range(len(self.segments)):
            if point in self.segments[i]:
                self.segments.insert(
                    i + 1, Segment(point, self.segments[i].end)
                )
                self.segments.insert(
                    i + 1, Segment(self.segments[i].start, point)
                )
                self.segments.pop(i)
                break

    def append_point(self, point):
        self.segments.append(Segment(self.segments[-1].end, point))

    def prepend_point(self, point):
        self.segments.insert(0, Segment(point, self.segments[0].start))

    def neighbours(self, point):
        if point not in self:
            return []
        for i in range(len(self.segments)):
            if point in self.segments[i]:
                alpha = self.segments[i].closest_alpha(point)
                if alpha < 1e-5 and i > 0:
                    return self.segments[i - 1].start, self.segments[i].end
                elif alpha > 1 - 1e-5 and i < len(self.segments) - 1:
                    return self.segments[i].start, self.segments[i + 1].end
                return self.segments[i].start, self.segments[i].end

    def __str__(self):
        return "OrderedPath:{}".format(self.segments)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        assert isinstance(other, OrderedPath)
        return self in other and other in self


def paths_to_polygons(paths):
    # the first step is to record all the intersections
    paths_intersection(paths)
    # then we have to loop through all the vertices
    all_vertices = []
    for path in paths:
        for vertex in path.vertices:
            if vertex not in all_vertices:
                all_vertices.append(vertex)
    segments = []
    for vertex in all_vertices:
        for path in paths:
            neigh = path.neighbours(vertex)
            for n in neigh:
                if n == vertex:
                    continue
                segments.append(Segment(vertex, n))

    done = []
    polygons = []
    for segment in segments:

        identifier = str(segment)
        if identifier in done:
            continue
        done.append(identifier)

        con = True
        for p in polygons:
            if segment in p:
                con = False
        if not con:
            continue

        print("segment", segment)
        polygon = segment_to_polygon(segment, paths, add_intersection=False)
        if polygon is not None and polygon not in polygons:
            polygons.append(polygon)
    return polygons


def paths_intersection(paths):
    if len(paths) == 1:
        return
    for i in range(len(paths)):
        for j in range(len(paths)):
            if i >= j:
                continue
            nodes = paths[i].intersect(paths[j])
            if nodes is None:
                continue
            for node in nodes:
                paths[i].insert_point(node)
                paths[j].insert_point(node)


def segment_to_polygon(segment, paths, clockwise=True, add_intersection=True):
    if add_intersection:
        paths_intersection(paths)

    if not np.any([segment in path for path in paths]):
        return None

    polygon = OrderedPath(segments=[segment])
    while polygon.end != polygon.start:
        neighbours = []
        current = polygon.end
        for path in paths:
            neighbours += path.neighbours(current)
        if current in neighbours:
            neighbours.remove(current)
        if polygon.start in neighbours and len(polygon) > len(polygon.start):
            polygon.append_point(polygon.start)
            return polygon
        neighbours = [point for point in neighbours if point not in polygon]
        if len(neighbours) == 0:
            return None
        angles = np.array(
            [
                (polygon.segments[-1].end - polygon.start).angle(
                    point - polygon.segments[-1].end
                )
                for point in neighbours
            ]
        )

        angles[angles > 180] = 0
        tokeep = np.argmax(angles)
        if angles[tokeep] > 180:
            return None
        polygon.append_point(neighbours[tokeep])


def get_layer_PD(inputs, layer_W, layer_b, layer_alpha, colors=None):

    h = inputs.dot(layer_W.T) + layer_b

    code = np.where(h > 0, 1, layer_alpha)

    code, indices = np.unique(code, return_index=True, axis=0)

    mus = np.einsum("nd,dk->nk", code, layer_W)
    print(
        2 * (layer_b * code).sum(1) + np.linalg.norm(mus, axis=1, ord=2) ** 2
    )
    radii = (
        2 * (layer_b * code).sum(1) + np.linalg.norm(mus, axis=1, ord=2) ** 2
    )
    radii -= radii.min()
    radii = np.sqrt(radii)

    if colors is not None:
        colors = colors[indices]
        return mus, radii, colors
    else:
        return mus, radii


def connected_components(graph):
    """
    represent the graph using an adjacency list, you can use this generator function (implementing BFS) to get all connected components:
    """
    seen = set()

    for root in range(len(graph)):
        if root not in seen:
            seen.add(root)
            component = []
            queue = deque([root])

            while queue:
                node = queue.popleft()
                component.append(node)
                for neighbor in graph[node]:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        queue.append(neighbor)
            yield component


def get_graph(paths, precision=5):
    # first transform a vertex to an int
    # and round to prevent errors
    vertex_to_int = {}
    count = 0
    for path in paths:
        for v in path.vertices[:-1]:
            vertex_to_int[tuple(np.round(v, precision))] = count
            count += 1

    nodes = []
    real_nodes = []
    graph = []
    for path in paths:
        for start, end in zip(path.vertices[:-2], path.vertices[1:-1]):
            cstart = vertex_to_int[tuple(np.round(start, precision))]
            cend = vertex_to_int[tuple(np.round(end, precision))]

            if cstart not in nodes:
                nodes.append(cstart)
                real_nodes.append(start)
                graph.append([])
                istart = len(nodes) - 1
            else:
                istart = nodes.index(cstart)

            if cend not in nodes:
                nodes.append(cend)
                real_nodes.append(end)
                graph.append([])
                iend = len(nodes) - 1
            else:
                iend = nodes.index(cend)

            graph[istart] += [iend]
            graph[iend] += [istart]
    return np.array(real_nodes), graph


def plot_polygon_search():
    # PATH
    p1 = OrderedPath(
        vertices=[
            Point(0, 0),
            Point(1, 0),
            Point(1, 1),
            Point(0, 1),
            Point(0, 0),
        ]
    )
    p2 = OrderedPath(vertices=[Point(1, 0), Point(1, 1),])
    p3 = OrderedPath(
        vertices=[
            Point(1, 1),
            Point(0, 1),
            Point(0, 0),
            Point(1, 0),
            Point(1, 1),
        ]
    )
    print(p1 == p3, p2 == p3, p2 in p3)

    # print(segment_to_polygon(Segment(Point(0, 0), Point(1, 0)), [p1]))
    # pp = paths_to_polygons([p1])
    # print("all", pp)

    p1 = OrderedPath(
        vertices=[Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1),]
    )
    p2 = OrderedPath(vertices=[Point(0.5, -2), Point(0.5, 2)])
    p3 = OrderedPath(vertices=[Point(0.0, 0.2), Point(1, 0.2)])
    p4 = OrderedPath(vertices=[Point(0.0, 0.0), Point(1, 1)])

    # print(paths_intersection([p1, p2]))
    # print(p1.vertices)
    # print(p2.vertices)
    # print(p1.neighbours(Point(0.5, 0)))
    # print(p2.neighbours(Point(0.5, 0)))
    import matplotlib.pyplot as plt

    colors = ["r", "k", "orange", "b", "g", "m", "y"]
    plt.subplot(211)
    for k, p in enumerate([p1, p2, p3, p4]):
        for seg in p:
            plt.plot(
                [seg.start[0], seg.end[0]],
                [seg.start[1], seg.end[1]],
                c=colors[k],
            )
    polygons = paths_to_polygons([p1, p2, p3, p4])
    for k, p in enumerate(polygons):
        plt.subplot(2, len(polygons), 1 + len(polygons) + k)
        for seg in p:
            plt.plot(
                [seg.start[0], seg.end[0]],
                [seg.start[1], seg.end[1]],
                c=colors[k],
                alpha=0.5,
            )
            plt.xlim([0, 1])
            plt.ylim([-2, 2])
    plt.show()
    print("all", paths_to_polygons([p1, p2, p3]))


if __name__ == "__main__":
    # POINTS
    print(Point(1, 1).angle())
    print(Point(1, -1).angle())

    # SEGMENTS
    s1 = Segment(Point(-1, -1), Point(1, 1))
    s2 = Segment(Point(-1, 1), Point(1, -1))
    # should be (0, 0)
    print(s1.intersect(s2))
    s1 = Segment(Point(0, 0), Point(1, 0))
    s2 = Segment(Point(0, 0), Point(0, 1))
    # should also be (0, 0)
    print(s1.intersect(s2))
    s1 = Segment(Point(0, 0), Point(1, 0))
    s2 = Segment(Point(0, 0.1), Point(0, 1))
    # should not be (0, 0)
    print(s1.intersect(s2))

    print(Point(1, 1).angle(Point(-1, 1)))
    print(Point(1, 1).angle(Point(-1, -1)))
    print(Point(1, 1).angle(Point(1, -1)))

    # PATH
    path = OrderedPath(vertices=[Point(0, 0), Point(1, 0), Point(1, 1)])
    otherpath = OrderedPath(
        vertices=[Point(0, 1), Point(0.5, -1), Point(1.4, 1)]
    )

    print(path.intersect(otherpath))
    print(path)
    path.insert_point(Point(0.5, 0))
    print(path)

    print(path.neighbours(Point(0.5, 0)))
    print(path.neighbours(Point(0.2, 0)))

    print(plot_polygon_search())
