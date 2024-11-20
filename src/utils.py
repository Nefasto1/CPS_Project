import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from src.Obstacle import Obstacle

import imageio

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

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]

    plt.close()

    return data

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
    ax.scatter(targets[:, 0], targets[:, 1], color="yellow", zorder=10)

    if len(target_list) > 0:
        # Plot the temporaneous target positions
        ax.scatter(target_list[:, 0], target_list[:, 1], color="black", zorder=10)

    # Show the plot
    plt.show()

def make_gif(frames, name, fps):
    imageio.mimsave(name, frames, fps=30)

def get_modules(directional_speeds):
    modules = directional_speeds**2
    modules = modules.sum(-1)
    modules = np.sqrt(modules)

    return modules
    
def get_directional(modules, thetas):
    x = modules * np.cos(thetas)
    y = modules * np.sin(thetas)

    return x, y

def window_mean(values, window_len=30):
    # Take the last window_len values minus one
    window = values[-window_len:]
    # Padding with zeros when not enough values
    window = [(0, 0)] * (window_len - len(window)) + window

    window = np.array(window)

    # Create the weigths and normalize them
    weight = np.arange(window_len).astype(np.float64)
    weight[:  len(weight)//3] *= 0.5
    weight[:2*len(weight)//3] *= 0.5
    weight /= weight.sum()

    # Multiply the speed values by the weight
    window[:, 0] = window[:, 0] * weight
    window[:, 1] = window[:, 1] * weight

    # Take the weighted sum
    return np.sum(window, 0)

def plot_inputs(inputs, references):
    ref_x, ref_y = references.transpose(2, 0, 1)
    
    plt.plot(np.arange(inputs.shape[1]), inputs.mean(0)[:, 0], label="x", zorder=3)
    plt.plot(np.arange(inputs.shape[1]), ref_x.mean(0), label="ref_x")
    plt.legend()
    plt.show()

    plt.plot(np.arange(inputs.shape[1]), inputs.mean(0)[:, 1], label="y", zorder=3)
    plt.plot(np.arange(inputs.shape[1]), ref_y.mean(0), label="ref_y")
    plt.legend()
    plt.show()
