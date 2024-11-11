import numpy as np
import matplotlib.pyplot as plt

### Separating Axis Theorem (SAT)
def get_axes(rect):
    # Get the axes perpendicular to each edge of the rectangle
    corners = rect.get_corners()
    axes = []
    for i in range(4):
        p1 = corners[i]              # Corner order = tr - tl - bl - br
        p2 = corners[(i + 1) % 4]    # Corner order = tl - bl - br - tr
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])  # Perpendicular vector
        norm = np.hypot(normal[0], normal[1])
        axes.append((normal[0] / norm, normal[1] / norm))
    return axes

def project(rect, axis):
    # Project each point of the rectangle onto the axis and return min and max
    dots = [(point[0] * axis[0] + point[1] * axis[1]) for point in rect.get_corners()]
    return min(dots), max(dots)

def overlap(min1, max1, min2, max2):
    # Check if projections overlap
    return max1 >= min2 and max2 >= min1

def rectangles_overlap(rect1, rect2):
    # Get all axes to check (both sets of rectangle edges)
    axes = get_axes(rect1) + get_axes(rect2)
    for axis in axes:
        min1, max1 = project(rect1, axis)
        min2, max2 = project(rect2, axis)
        if not overlap(min1, max1, min2, max2):
            return False  # Separating axis found, no overlap
    return True  # No separating axis found, rectangles overlap

def draw(car, obstacles):
    image_size = 1000
    fig = plt.figure(figsize=(image_size / 100, image_size / 100), dpi=100)
    ax = plt.gca()
    ax.axis('off')
    
    ax.set_xlim(0, image_size)
    ax.set_ylim(0, image_size)
    
    car.draw(ax)
    
    corners = car.get_corners()
    plt.scatter(*corners[0], color="blue",   zorder=5) # tl
    plt.scatter(*corners[1], color="green",  zorder=5) # bl
    plt.scatter(*corners[2], color="purple", zorder=5) # br
    plt.scatter(*corners[3], color="black",  zorder=5) # tr

    for obstacle in obstacles:
        obstacle.draw(ax)
        
        corners = obstacle.get_corners()
        plt.scatter(*corners[0], color="blue",   zorder=5) # tl
        plt.scatter(*corners[1], color="green",  zorder=5) # bl
        plt.scatter(*corners[2], color="purple", zorder=5) # br
        plt.scatter(*corners[3], color="black",  zorder=5) # tr
    plt.show()