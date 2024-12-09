import numpy as np

def bounding_rectangle(vertices):
    """
    Computes the bounding rectangle for a quadrilateral.
    
    Parameters:
        vertices (list of tuples): Vertices of the quadrilateral.
    
    Returns:
        list of tuples: Vertices of the bounding rectangle.
    """
    x_coords = [v[0] for v in vertices]
    y_coords = [v[1] for v in vertices]
    xmin, xmax = min(x_coords), max(x_coords)
    ymin, ymax = min(y_coords), max(y_coords)
    
    return [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]

def is_point_in_triangle(point, triangle):
    """
    Check if a point is inside a triangle using barycentric coordinates.
    
    Args:
        point (tuple): (x, y) coordinates of the point to check
        triangle (list): List of 3 tuples, each representing vertex coordinates [(x1,y1), (x2,y2), (x3,y3)]
    
    Returns:
        bool: True if point is inside the triangle, False otherwise
    """
    def compute_barycentric_coordinates(pt, v1, v2, v3):
        """
        Compute barycentric coordinates of a point with respect to a triangle.
        
        Args:
            pt (tuple): Point coordinates
            v1, v2, v3 (tuple): Vertex coordinates of the triangle
        
        Returns:
            tuple: Barycentric coordinates (u, v, w)
        """
        pt = np.array(pt)
        v1, v2, v3 = np.array(v1), np.array(v2), np.array(v3)
        
        # Vectorized area computation
        triangle_area = np.abs(np.cross(v2 - v1, v3 - v1)) / 2
        
        # Areas of sub-triangles
        area1 = np.abs(np.cross(pt - v2, v3 - v2)) / 2
        area2 = np.abs(np.cross(v1 - pt, v3 - v1)) / 2
        area3 = np.abs(np.cross(v1 - v2, pt - v2)) / 2
        
        # Compute barycentric coordinates
        u = area1 / triangle_area
        v = area2 / triangle_area
        w = area3 / triangle_area
        
        return u, v, w
    
    # Compute barycentric coordinates
    u, v, w = compute_barycentric_coordinates(point, triangle[0], triangle[1], triangle[2])
    
    # Point is inside if all barycentric coordinates are between 0 and 1 (inclusive)
    return 0 <= u <= 1 and 0 <= v <= 1 and 0 <= w <= 1 and np.abs(u + v + w - 1) < 1e-10

def is_point_in_parallelogram(point, box):
    """
    Check if a point is inside a parallelogram.
    
    Args:
        point (tuple): (x, y) coordinates of the point to check
        box (list): List of 4 tuples, each representing vertex coordinates [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    
    Returns:
        bool: True if point is inside the parallelogram, False otherwise
    """
    # Check if the point is in one of the two triangles of the parallelogram
    return is_point_in_triangle(point, [box[0], box[1], box[2]]) or is_point_in_triangle(point, [box[0], box[2], box[3]])

def sample_in_parallelogram(box):
    """
    Sample a point uniformly inside a parallelogram.
    
    Args:
        box (list): List of 4 tuples, each representing vertex coordinates [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    
    Returns:
        tuple: (x, y) coordinates of the sampled point
    """

    # Compute bounding box
    rect = bounding_rectangle(box)
    x_min, y_min = rect[0]
    x_max, y_max = rect[2]    
    # Keep sampling until a point inside the parallelogram is found

    x = np.random.uniform(x_min, x_max)
    y = np.random.uniform(y_min, y_max)
    while not is_point_in_parallelogram((x, y), box):
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
    
    return [x, y]