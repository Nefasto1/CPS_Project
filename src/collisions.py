import numpy as np
import math

def get_cross(width, height):
    """
    Function to retrieve the cross position at 150 pixels of distance from the center, given the height and the width

    Parameters
    ----------
    width: The object width
    height: The object height

    Returns
    -------
    cross: The list of cross positions
    """
    return np.array([ (+ width + 150, 0), 
                      (- width - 150, 0),
                      (0, + height + 150),
                      (0, - height - 150)])

def get_corners(width, height):
    """
    Function to retrieve the corners position at 150 pixels of distance from the center, given the height and the width

    Parameters
    ----------
    width: The object's width
    height: The object's height

    Returns
    -------
    corners: The list of corners positions
    """
    return np.array([ (+ width + 150, + height), 
                      (+ width,      + height + 150), 
                      (- width - 150, + height),
                      (- width,      + height + 150),
                      (- width - 150, - height),
                      (- width,      - height - 150),
                      (+ width + 150, - height), 
                      (+ width,      - height - 150),
                    ])

def rotate(coords, theta):
    """
    Rotate the coordinates of an object given the coordinates and the angle

    Parameters
    ----------
    coords: The object's coordninates
    theta: The object's angle

    Returns
    -------
    rotated_coords: The rotated coordinates
    """
    x = coords[:, 0] * math.cos(math.radians(theta)) \
      - coords[:, 1] * math.sin(math.radians(theta))

    y = coords[:, 0] * math.sin(math.radians(theta)) \
      + coords[:, 1] * math.cos(math.radians(theta))

    return np.hstack((x[:, np.newaxis], y[:, np.newaxis]))

def get_rotated(coords, width, height, theta):
    """
    Rotate the coordinates of an object

    Parameters
    ----------
    coords: The object's coordninates
    width: The object's width
    height: The object's height
    theta: The object's angle

    Returns
    -------
    corners: The rotated coordinates
    """
    # Get the cross positions
    corners = get_cross(width, height)
    # Get the corners positions
    corners = np.vstack((corners, get_corners(width, height)))
    # Get rotation shift for the positions
    corners = rotate(corners, theta)

    # Rotate the coordinates
    corners[:, 0] = coords[0] + corners[:, 0] 
    corners[:, 1] = coords[1] + corners[:, 1] 
    
    return corners

def get_distances(car_coords, obstacles):
    """
    Function to determine the distance between the car and a list of obstacles

    Parameters
    ----------
    car_coords: The car's coordinates
    obstacles: The list of initialized Obstacles

    Returns
    -------
    distances: The list with the car-obstacles distances
    """
    # Retrieve the obstacles coordinates
    obstacle_coords = [obstacle.get_coords() for obstacle in obstacles]
    obstacle_corner = [obstacle.get_corners() for obstacle in obstacles]
    obstacle_coords = np.array(obstacle_coords)
    obstacle_corner = np.array(obstacle_corner)

    # Take the absolute difference between car and obstacles
    distances = np.abs(obstacle_coords - car_coords)
    corner_dist = np.abs(obstacle_corner - car_coords)
    
    return distances, corner_dist
    
def get_new_targets(car_state, obstacles, target):
    """
    Function to get new temporaneous targets if obstacles are detected nearby

    Parameters
    ----------
    car_state: The car's state
    obstacles: The list of initialized Obstacles
    target: The target state

    Returns
    -------
    new_target: The new target state
    counter: The number of iteration with original target to do after
    """
    # Initialize the values to the original target
    new_target, counter = target, 1

    # Retrieve the car's coordinates and angle
    car_coords = car_state[:2]
    car_theta  = np.arctan2(car_coords[1], car_coords[0])

    # Retrieve the distances and modules between car and obstacles' corner
    distances, real_distances = get_distances(car_coords, obstacles)
    modules   = np.sqrt((distances**2).sum(1))

    # Determine the angle between the car and the obstacles
    thetas = np.arctan2(distances[:, 1], distances[:, 0])
    thetas = np.abs(np.rad2deg(thetas - car_theta)) % 360

    # Find the closest obstacle corner
    minimum          = np.argmin(modules)
    minimum_obstacle = obstacles[minimum]

    real_distances = real_distances.reshape(-1, 2)
    real_distances = np.sqrt((real_distances**2).sum(1))
    real_distances = real_distances.min()
    
    # If the closest is below a threshold and is in direction between the target and car determine a temporaneous target
    # Otherwise return the original one
    if modules[minimum] < 200 and thetas[minimum] < 45:
        # Retrive the closest obstacle informations
        obstacle_coords = minimum_obstacle.get_coords()
        obstacle_theta  = minimum_obstacle.theta
        obstacle_width  = minimum_obstacle.width
        obstacle_height = minimum_obstacle.height

        # Retrive the slightly distant coordinates to the obstacle
        new_candidates = get_rotated(obstacle_coords, obstacle_width, obstacle_height, obstacle_theta)

        # Determine the distances between car and slightly distant coordinates to the obstacle
        distances = new_candidates - car_coords

        car_obs   = obstacle_coords - car_coords
        car_obs_t = np.arctan2(car_obs[1], car_obs[0])
        thetas    = np.arctan2(distances[:, 1], distances[:, 0])
        diff      = thetas - car_obs_t
        diff      = np.rad2deg(np.abs(diff))
        
        distances = distances**2
        distances = np.sqrt(distances.sum(1))
        distances[diff < 25] = 1e16

        # Find the most suitable temporaneous target
        minimum = np.argmin(distances)
        new_target = new_candidates[minimum]
        new_target = np.hstack((new_target, [0, 0]))
        
        # Use the standard target for at least 3 times before check for a new temporaneous target
        # Remove the collapse to stay in the temporaneous location infinitely
        counter = 3 if np.sqrt((car_obs**2).sum()) > 75 else 1
        
    return new_target, real_distances, counter
    
# Check for overlap using Separating Axis Theorem
def check_overlap(rect1, rect2):
    def project_polygon(axis, polygon):
        projections = np.dot(polygon, axis)
        return min(projections), max(projections)
    
    def overlap_on_axis(axis, rect1, rect2):
        min1, max1 = project_polygon(axis, rect1)
        min2, max2 = project_polygon(axis, rect2)
        return not (max1 < min2 or max2 < min1)

    # Combine edges from both rectangles
    edges = np.vstack([rect1, rect2])
    axes = np.diff(edges, axis=0)
    axes = np.vstack([axes, np.roll(axes, shift=1, axis=0)])  # Orthogonal axes

    # Normalize axes
    axes = axes / np.linalg.norm(axes, axis=1)[:, None]
    
    # Check for overlap on all axes
    return all(overlap_on_axis(axis, rect1, rect2) for axis in axes)

def check_collision(car, obstacles):
    car_corners = car.get_coords()[None, :].repeat(4, axis=0)
    
    for obstacle in obstacles:
        obstacle_corners = obstacle.rotate_corners()
        x, y             = obstacle.get_coords()
        obstacle_corners[:, 0] += x
        obstacle_corners[:, 1] += y
        if check_overlap(car_corners, obstacle_corners):
            return True
    return False