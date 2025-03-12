"""
Geometry utility functions for airspace network planning.
"""

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
    inner_circle = (center, inner_radius)
    outer_circle = (center, outer_radius)

    # Get intersection points with outer circle
    outer_intersections = line_circle_intersection(line, outer_circle)

    # If less than 2 intersections, no multiple segments in ring
    if len(outer_intersections) < 2:
        return False

    # Check if the line already has nodes in the outer ring
    # Find the node indices for the line
    p1, p2 = line
    p1_idx = None
    p2_idx = None

    for i, node in enumerate(nodes):
        if np.allclose(node, p1):
            p1_idx = i
        if np.allclose(node, p2):
            p2_idx = i

    if p1_idx is None or p2_idx is None:
        return False  # New nodes being proposed, no existing connections yet

    # Check for connected nodes in the outer ring
    for edge in edges:
        if p1_idx in edge or p2_idx in edge:
            # Check if the connected node is in the outer ring
            connected_idx = edge[0] if edge[1] in (p1_idx, p2_idx) else edge[1]
            connected_point = tuple(nodes[connected_idx])

            # Check if connected point is in outer ring but not in inner circle
            in_outer = is_point_in_circle(connected_point, outer_circle)
            in_inner = is_point_in_circle(connected_point, inner_circle)

            if in_outer and not in_inner:
                return True  # Found another segment in the outer ring

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

def is_line_segment_valid(
    start: Tuple[float, float],
    end: Tuple[float, float],
    adi_zones: List[Dict],
    existing_edges: List[Tuple[int, int]],
    nodes: np.ndarray,
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
        max_angle_deg: Maximum allowed angle in degrees

    Returns:
        Tuple of (is_valid, reason, penalty)
    """
    line = (start, end)

    # Check if line crosses any inner ADI zones (forbidden)
    for zone in adi_zones:
        crosses_inner, crosses_outer = does_line_cross_adi_zone(line, zone)

        if crosses_inner:
            return False, "Crosses inner ADI zone", 0.0

        # Check for zigzag paths in outer ring
        if crosses_outer:
            if does_line_intersect_multiple_segments_in_outer_ring(line, zone, existing_edges, nodes):
                return False, "Creates zigzag in outer ADI ring", 0.0

    # Check angle constraints with existing segments
    start_idx = None
    end_idx = None

    for i, node in enumerate(nodes):
        if np.allclose(node, start):
            start_idx = i
        if np.allclose(node, end):
            end_idx = i

    if start_idx is not None:
        # Check angles with existing edges from start point
        for edge in existing_edges:
            if start_idx in edge:
                other_idx = edge[0] if edge[1] == start_idx else edge[1]
                other_point = tuple(nodes[other_idx])

                if other_point != end:  # Avoid checking the proposed edge itself
                    line1 = (other_point, start)
                    line2 = (start, end)
                    try:
                        angle = angle_between_lines(line1, line2)
                        if angle < max_angle_deg:
                            return False, f"Angle too sharp ({angle:.1f}°)", 0.0
                    except ValueError:
                        # Skip if the lines don't share an endpoint
                        pass

    if end_idx is not None:
        # Check angles with existing edges from end point
        for edge in existing_edges:
            if end_idx in edge:
                other_idx = edge[0] if edge[1] == end_idx else edge[1]
                other_point = tuple(nodes[other_idx])

                if other_point != start:  # Avoid checking the proposed edge itself
                    line1 = (other_point, end)
                    line2 = (end, start)
                    try:
                        angle = angle_between_lines(line1, line2)
                        if angle < max_angle_deg:
                            return False, f"Angle too sharp ({angle:.1f}°)", 0.0
                    except ValueError:
                        # Skip if the lines don't share an endpoint
                        pass

    return True, "", 0.0