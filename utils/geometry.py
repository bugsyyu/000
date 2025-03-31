"""
Geometry utility functions for airspace network planning.
"""
import warnings
import numpy as np
from typing import Tuple, List, Union, Dict, Optional

def distance_point_to_point(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two points.

    Args:
        p1: First point (x1, y1)
        p2: Second point (x2, y2)

    Returns:
        Euclidean distance between the points
    """
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def angle_between_lines(line1: Tuple[Tuple[float, float], Tuple[float, float]],
                        line2: Tuple[Tuple[float, float], Tuple[float, float]]) -> float:
    """
    Calculate the angle between two lines in degrees.

    Args:
        line1: First line as ((x1, y1), (x2, y2))
        line2: Second line as ((x3, y3), (x4, y4))

    Returns:
        Angle in degrees (0-180)
    """
    # Extract points
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2

    # Check if the lines share a point
    if (x2, y2) != (x3, y3):
        raise ValueError("Lines must share an endpoint to calculate angle")

    # Calculate vectors
    v1 = np.array([x2 - x1, y2 - y1])
    v2 = np.array([x4 - x3, y4 - y3])

    # Normalize vectors
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)

    # Calculate angle in radians
    dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)

    # Convert to degrees
    angle_deg = np.degrees(angle_rad)

    return angle_deg

def line_circle_intersection(line: Tuple[Tuple[float, float], Tuple[float, float]],
                             circle: Tuple[Tuple[float, float], float]) -> List[Tuple[float, float]]:
    """
    Calculate intersection points between a line segment and a circle.

    Args:
        line: Line segment as ((x1, y1), (x2, y2))
        circle: Circle as ((center_x, center_y), radius)

    Returns:
        List of intersection points
    """
    # Extract points
    (x1, y1), (x2, y2) = line
    (center_x, center_y), radius = circle

    # Convert line to parametric form: P = P1 + t * (P2 - P1)
    dx = x2 - x1
    dy = y2 - y1

    # Translate circle center to origin
    x1_t = x1 - center_x
    y1_t = y1 - center_y

    # Calculate coefficients for quadratic equation
    a = dx**2 + dy**2
    b = 2 * (x1_t * dx + y1_t * dy)
    c = x1_t**2 + y1_t**2 - radius**2

    # Calculate discriminant
    discriminant = b**2 - 4 * a * c

    # No intersection
    if discriminant < 0:
        return []

    # Tangent (one intersection)
    if discriminant == 0:
        t = -b / (2 * a)
        # Check if intersection is on the line segment
        if 0 <= t <= 1:
            x = x1 + t * dx
            y = y1 + t * dy
            return [(x, y)]
        return []

    # Two intersections
    t1 = (-b + np.sqrt(discriminant)) / (2 * a)
    t2 = (-b - np.sqrt(discriminant)) / (2 * a)

    intersections = []

    # Check if first intersection is on the line segment
    if 0 <= t1 <= 1:
        x = x1 + t1 * dx
        y = y1 + t1 * dy
        intersections.append((x, y))

    # Check if second intersection is on the line segment
    if 0 <= t2 <= 1:
        x = x1 + t2 * dx
        y = y1 + t2 * dy
        intersections.append((x, y))

    return intersections

def is_point_in_circle(point: Tuple[float, float], circle: Tuple[Tuple[float, float], float]) -> bool:
    """
    Check if a point is inside a circle.

    Args:
        point: Point as (x, y)
        circle: Circle as ((center_x, center_y), radius)

    Returns:
        True if the point is inside the circle, False otherwise
    """
    (x, y) = point
    (center_x, center_y), radius = circle

    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)

    return distance <= radius

def does_line_cross_circle(line: Tuple[Tuple[float, float], Tuple[float, float]],
                           circle: Tuple[Tuple[float, float], float]) -> bool:
    """
    Check if a line segment intersects with a circle.

    Args:
        line: Line segment as ((x1, y1), (x2, y2))
        circle: Circle as ((center_x, center_y), radius)

    Returns:
        True if the line intersects the circle, False otherwise
    """
    # Check if either endpoint is inside the circle
    if is_point_in_circle(line[0], circle) or is_point_in_circle(line[1], circle):
        return True

    # Check for intersections
    intersections = line_circle_intersection(line, circle)
    return len(intersections) > 0

def does_line_cross_adi_zone(line: Tuple[Tuple[float, float], Tuple[float, float]],
                             adi_zone: Dict) -> Tuple[bool, bool]:
    """
    Check if a line segment crosses the inner or outer radius of an ADI zone.

    Args:
        line: Line segment as ((x1, y1), (x2, y2))
        adi_zone: ADI zone parameters with center_x, center_y, radius, epsilon

    Returns:
        Tuple of (crosses_inner, crosses_outer)
    """
    center = (adi_zone['center_x'], adi_zone['center_y'])
    inner_radius = adi_zone['radius']
    outer_radius = adi_zone['epsilon']

    inner_circle = (center, inner_radius)
    outer_circle = (center, outer_radius)

    crosses_inner = does_line_cross_circle(line, inner_circle)
    crosses_outer = does_line_cross_circle(line, outer_circle)

    return crosses_inner, crosses_outer

def does_line_intersect_multiple_segments_in_outer_ring(
    line: Tuple[Tuple[float, float], Tuple[float, float]],
    adi_zone: Dict,
    edges: List[Tuple[int, int]],
    nodes: np.ndarray
) -> bool:
    """
    Check if a line creates multiple segments within the outer ring of an ADI zone.
    This is to prevent zigzag paths within the ADI zone's outer ring.

    Args:
        line: Line segment as ((x1, y1), (x2, y2))
        adi_zone: ADI zone parameters
        edges: Existing edges as list of (node1, node2) indices
        nodes: Node coordinates as array of (x, y)

    Returns:
        True if the line would create multiple segments in the outer ring, False otherwise
    """
    center = (adi_zone['center_x'], adi_zone['center_y'])
    inner_radius = adi_zone['radius']
    outer_radius = adi_zone['epsilon']

    # Check if the line intersects the outer ring
    outer_intersections = line_circle_intersection(line, (center, outer_radius))
    if len(outer_intersections) < 2:
        return False

    # Check if the line already has nodes in the outer ring
    (x1, y1), (x2, y2) = line

    # find start/end indices
    p1_idx = None
    p2_idx = None
    for i, node in enumerate(nodes):
        if np.allclose(node, (x1, y1)):
            p1_idx = i
        if np.allclose(node, (x2, y2)):
            p2_idx = i

    if p1_idx is None or p2_idx is None:
        return False  # new node, skip

    # see if from p1_idx or p2_idx there is an existing edge with node in outer ring but not inner
    for edge in edges:
        if p1_idx in edge or p2_idx in edge:
            other_idx = edge[0] if edge[1] in (p1_idx, p2_idx) else edge[1]
            other_point = tuple(nodes[other_idx])
            # check if in outer ring but not inner
            in_outer = is_point_in_circle(other_point, (center, outer_radius))
            in_inner = is_point_in_circle(other_point, (center, inner_radius))
            if in_outer and not in_inner:
                return True
    return False

def calculate_path_length(path: List[Tuple[float, float]]) -> float:
    """
    Calculate the total length of a path.

    Args:
        path: List of points [(x1, y1), (x2, y2), ...]

    Returns:
        Total length of the path
    """
    if len(path) < 2:
        return 0.0

    total_length = 0.0
    for i in range(len(path) - 1):
        total_length += distance_point_to_point(path[i], path[i+1])
    return total_length

def get_danger_zone_penalty(line: Tuple[Tuple[float, float], Tuple[float, float]],
                            danger_zones: List[Dict]) -> float:
    """
    Calculate penalty for crossing danger zones.

    Args:
        line: Line segment as ((x1, y1), (x2, y2))
        danger_zones: List of danger zone parameters

    Returns:
        Total penalty for crossing danger zones
    """
    total_penalty = 0.0
    for zone in danger_zones:
        center = (zone['center_x'], zone['center_y'])
        radius = zone['radius']
        circle = (center, radius)

        if does_line_cross_circle(line, circle):
            total_penalty += zone['threat_level']

    return total_penalty


def _on_segment(p: Tuple[float, float], q: Tuple[float, float], r: Tuple[float, float]) -> bool:
    """
    Given three colinear points p, q, r, check if point q lies on segment pr.
    """
    if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
        q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
        return True
    return False

def _orientation(p: Tuple[float, float],
                 q: Tuple[float, float],
                 r: Tuple[float, float]) -> int:
    """
    Find orientation of ordered triplet (p, q, r).
    Returns:
     0 -> p, q, r are colinear
     1 -> Clockwise
     2 -> Counterclockwise
    """
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if abs(val) < 1e-12:
        return 0
    return 1 if val > 0 else 2

def do_segments_intersect_excl_endpoints(
    line1: Tuple[Tuple[float, float], Tuple[float, float]],
    line2: Tuple[Tuple[float, float], Tuple[float, float]]
) -> bool:
    """
    Check if two line segments intersect, ignoring intersection at a shared endpoint.
    (i.e. if they share an endpoint exactly, that does NOT count as an intersection.)

    Return True if they properly intersect in the interior, or if they overlap partially but not
    just at a single shared endpoint.

    We use the classic orientation-based approach, plus a check for colinearity, etc.
    """
    p1, q1 = line1
    p2, q2 = line2

    # if any endpoints are identical, we skip counting that as an intersection
    # (but if they share exactly one endpoint, that's allowed adjacency, not crossing).
    shared_endpoints = 0
    if np.allclose(p1, p2) or np.allclose(p1, q2):
        shared_endpoints += 1
    if np.allclose(q1, p2) or np.allclose(q1, q2):
        shared_endpoints += 1

    # Next, do the orientation checks
    o1 = _orientation(p1, q1, p2)
    o2 = _orientation(p1, q1, q2)
    o3 = _orientation(p2, q2, p1)
    o4 = _orientation(p2, q2, q1)

    # General case
    if o1 != o2 and o3 != o4:
        # If they share only one endpoint, let's see if that's the only intersection
        if shared_endpoints == 1:
            # They do intersect, but exactly at the shared endpoint => not considered crossing
            return False
        return True

    # Special Cases
    # p1, q1, p2 are colinear and p2 lies on segment p1q1
    if o1 == 0 and _on_segment(p1, p2, q1):
        # check if that's a "shared endpoint" scenario or actual overlap
        if np.allclose(p2, p1) or np.allclose(p2, q1):
            # It's colinear, but it's exactly the shared endpoint => ignore
            return False
        return True

    # p1, q1, q2 are colinear and q2 lies on segment p1q1
    if o2 == 0 and _on_segment(p1, q2, q1):
        if np.allclose(q2, p1) or np.allclose(q2, q1):
            return False
        return True

    # p2, q2, p1 are colinear and p1 lies on segment p2q2
    if o3 == 0 and _on_segment(p2, p1, q2):
        if np.allclose(p1, p2) or np.allclose(p1, q2):
            return False
        return True

    # p2, q2, q1 are colinear and q1 lies on segment p2q2
    if o4 == 0 and _on_segment(p2, q1, q2):
        if np.allclose(q1, p2) or np.allclose(q1, q2):
            return False
        return True

    return False

def is_line_segment_valid(
        start: Tuple[float, float],
        end: Tuple[float, float],
        adi_zones: List[Dict],
        existing_edges: List[Tuple[int, int]],
        nodes: np.ndarray,
        node_types: List[int] = None,  # 可选参数
        airport_indices: List[int] = None,  # 可选参数
        max_angle_deg: float = 80.0
) -> Tuple[bool, str, float]:
    """
    Check if a line segment is valid according to all constraints.

    Args:
        start: Start point (x1, y1)
        end: End point (x2, y2)
        adi_zones: List of ADI zone parameters
        existing_edges: List of existing edges as (node1_idx, node2_idx)
        nodes: Array of node coordinates
        node_types: List of node types (0: frontline, 1: airport, 2: common, 3: outlier)
        airport_indices: List of airport node indices
        max_angle_deg: Maximum allowed angle in degrees

    Returns:
        Tuple of (is_valid, reason, penalty)
    """
    # 如果缺少关键参数，发出警告
    if node_types is None or airport_indices is None:
        warnings.warn(
            "node_types 或 airport_indices 未提供，将忽略前沿点/机场连线约束和机场禁飞区约束。"
            "请提供这些参数以确保所有约束条件都被检查。",
            UserWarning
        )

    line = (start, end)

    # 找到起点和终点的索引
    start_idx = None
    end_idx = None
    for i, node in enumerate(nodes):
        if np.allclose(node, start):
            start_idx = i
        if np.allclose(node, end):
            end_idx = i

    # 如果提供了节点类型和机场索引，则执行额外约束检查
    if node_types is not None and airport_indices is not None and start_idx is not None and end_idx is not None:
        # 约束1和2：禁止前沿点之间或机场之间直接连线
        if (node_types[start_idx] == 0 and node_types[end_idx] == 0):
            return False, "前沿点之间禁止直接连线", 0.0

        if (node_types[start_idx] == 1 and node_types[end_idx] == 1):
            return False, "机场之间禁止直接连线", 0.0

        # 约束3：机场周围20km范围内是禁飞区
        for airport_idx in airport_indices:
            airport_point = tuple(nodes[airport_idx])

            # 跳过如果起点或终点就是机场本身
            if start_idx == airport_idx or end_idx == airport_idx:
                continue

            # 检查线段是否穿过机场的20km禁飞区
            airport_no_fly_zone = (airport_point, 20.0)  # 20km半径
            if does_line_cross_circle(line, airport_no_fly_zone):
                return False, f"线段穿过机场{airport_idx}的禁飞区", 0.0

    # 检查是否穿过任何内部ADI区域（禁止）
    for zone in adi_zones:
        crosses_inner, crosses_outer = does_line_cross_adi_zone(line, zone)
        if crosses_inner:
            return False, "穿过内部ADI区域", 0.0

        # 防止在外圈之字形
        if crosses_outer:
            if does_line_intersect_multiple_segments_in_outer_ring(line, zone, existing_edges, nodes):
                return False, "在外部ADI环形成之字形路径", 0.0

    # 新增：检查是否和已有边交叉(除端点重合外)
    for (e1, e2) in existing_edges:
        p1 = tuple(nodes[e1])
        p2 = tuple(nodes[e2])
        if do_segments_intersect_excl_endpoints(line, (p1, p2)):
            return False, "与已有边发生交叉", 0.0

    # 检查与现有线段的角度约束
    if start_idx is not None:
        # 检查与从起点出发的现有边的角度
        for edge in existing_edges:
            if start_idx in edge:
                other_idx = edge[0] if edge[1] == start_idx else edge[1]
                other_point = tuple(nodes[other_idx])

                if other_point != end:  # 避免检查提议的边本身
                    line1 = (other_point, start)
                    line2 = (start, end)
                    try:
                        angle = angle_between_lines(line1, line2)
                        if angle < max_angle_deg:
                            return False, f"角度过小 ({angle:.1f}°)", 0.0
                    except ValueError:
                        pass

    if end_idx is not None:
        # 检查与从终点出发的现有边的角度
        for edge in existing_edges:
            if end_idx in edge:
                other_idx = edge[0] if edge[1] == end_idx else edge[1]
                other_point = tuple(nodes[other_idx])

                if other_point != start:  # 避免检查提议的边本身
                    line1 = (other_point, end)
                    line2 = (end, start)
                    try:
                        angle = angle_between_lines(line1, line2)
                        if angle < max_angle_deg:
                            return False, f"角度过小 ({angle:.1f}°)", 0.0
                    except ValueError:
                        pass

    return True, "", 0.0
