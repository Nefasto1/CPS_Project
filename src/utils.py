import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from src.Obstacle import Obstacle

def draw(car, obstacles):
    """
    Function to draw the current state of the system

    Parameters
    ----------
    car: The initialized car object
    obstacles: The list of initialized obstacles
    """
    # Initialize the plot
    image_size = 1000
    fig = plt.figure(figsize=(image_size / 100, image_size / 100), dpi=100)
    ax = plt.gca()
    ax.axis('off')
    
    ax.set_xlim(0, image_size)
    ax.set_ylim(0, image_size)

    # Draw the car
    car.draw(ax)

    # Draw the car's corner
    corners = car.get_corners()
    plt.scatter(*corners[0], color="blue",   zorder=5) # top-left corner
    plt.scatter(*corners[1], color="green",  zorder=5) # bottom-left corner
    plt.scatter(*corners[2], color="purple", zorder=5) # bottom-right corner
    plt.scatter(*corners[3], color="black",  zorder=5) # top-right corner

    # Draw the obstacles and relatives corners
    for obstacle in obstacles:
        obstacle.draw(ax)
        
        corners = obstacle.get_corners()
        plt.scatter(*corners[0], color="blue",   zorder=5) # top-left corner
        plt.scatter(*corners[1], color="green",  zorder=5) # bottom-left corner
        plt.scatter(*corners[2], color="purple", zorder=5) # bottom-right corner
        plt.scatter(*corners[3], color="black",  zorder=5) # top-right corner

    # Show the plot
    plt.show()
    

def draw_trajectories(obstacles, start, targets, states, target_list):
    """
    Function to draw the trajectories of the simulations

    Parameters
    ----------
    obstacles: The list of obstacles' informations
    start: The car start informations
    targets: The car target informations
    states: The car trajectories
    target_list: The list of temporaneous targets due to the obstacles
    """
    # Initialize the Obstacles
    obstacles_draw = np.array([Obstacle(coords, angle) for coords, angle in obstacles])

    # Initialize the plot
    image_size = 1000
    fig = plt.figure(figsize=(image_size / 100, image_size / 100), dpi=100)
    ax = plt.gca()
    ax.axis('off')
    
    ax.set_xlim(0, image_size)
    ax.set_ylim(0, image_size)

    # Plot the trajectories
    for state in states:
        ax.plot(state[:, 0], state[:, 1], color="purple")

    # Plot the starting position
    ax.scatter(start[0][0], start[0][1], color="green", zorder=10)

    # Plot the obstacles and relative detection boundings
    for obstacle in obstacles_draw:
        obstacle.draw(ax)
        ax.add_patch(Circle(obstacle.get_coords(), 200, lw=2, facecolor="None", edgecolor="red", zorder=10))

    # Plot the target position
    ax.scatter(targets[:, 0], targets[:, 1], color="green", zorder=10)

    # Plot the temporaneous target positions
    ax.scatter(target_list[:, 0], target_list[:, 1], color="yellow", zorder=10)

    # Show the plot
    plt.show()