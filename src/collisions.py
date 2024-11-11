import numpy as np
import math

def get_cross(width, height):
    return np.array([ (+ width + 150, 0), 
                      (- width - 150, 0),
                      (0, + height + 150),
                      (0, - height - 150)])

def get_corners(width, height):
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
    x = coords[:, 0] * math.cos(math.radians(theta)) \
      - coords[:, 1] * math.sin(math.radians(theta))

    y = coords[:, 0] * math.sin(math.radians(theta)) \
      + coords[:, 1] * math.cos(math.radians(theta))

    return np.hstack((x[:, np.newaxis], y[:, np.newaxis]))

def get_rotated(coords, width, height, theta):
    corners = get_cross(width, height)
    corners = np.vstack((corners, get_corners(width, height)))
    # Rotate the corners
    corners = rotate(corners, theta)

    corners[:, 0] = coords[0] + corners[:, 0] 
    corners[:, 1] = coords[1] + corners[:, 1] 
    
    return corners

def get_distances(car_coords, obstacles):
    obstacle_coords = [obstacle.get_coords() for obstacle in obstacles]
    obstacle_coords = np.array(obstacle_coords)

    distances = np.abs(obstacle_coords - car_coords)
    return distances
    
def get_new_targets(car_state, obstacles, target):
    new_target, counter = target, 1
    
    car_coords = car_state[:2]
    car_theta  = np.arctan2(car_coords[1], car_coords[0])
    
    distances = get_distances(car_coords, obstacles)
    modules   = np.sqrt((distances**2).sum(1))

    thetas = np.arctan2(distances[:, 1], distances[:, 0])
    thetas = np.abs(np.rad2deg(thetas - car_theta)) % 360

    minimum              = np.argmin(modules)
    minimum_obstacle     = obstacles[minimum]

    if modules[minimum] < 200 and thetas[minimum] < 45:
        obstacle_coords = minimum_obstacle.get_coords()
        obstacle_theta  = minimum_obstacle.theta
        obstacle_width  = minimum_obstacle.width
        obstacle_height = minimum_obstacle.height

        new_candidates = get_rotated(obstacle_coords, obstacle_width, obstacle_height, obstacle_theta)

        distances = new_candidates - car_coords
        distances = distances**2
        distances = np.sqrt(distances.sum(1))

        minimum = np.argmin(distances)
        new_target = new_candidates[minimum]
        new_target = np.hstack((new_target, [0, 0]))
        counter = 3
        
        
    return new_target, counter