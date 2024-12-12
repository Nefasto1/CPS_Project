import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from matplotlib.patches import Circle

from src.Car import Car
from src.Obstacle import Obstacle

def draw(states, thetas, obstacle_list):
    """
    Function to draw the current state of the system

    WARNING: Requires a LOT of RAM

    Parameters
    ----------
    states: car states
    thetas: car thetas
    obstacle_list: The list of obstacles
    """
    # Initialize the plot
    image_size = 1000
    fig = plt.figure(figsize=(image_size / 100, image_size / 100), dpi=100)
    
    def animate(i):
        print(i)
        ax = plt.gca()
        plt.cla()
        ax.axis('off')
        
        ax.set_xlim(0, image_size)
        ax.set_ylim(0, image_size)

        car       = Car(states[i, :2], states[i, 2:], thetas[i], noise=noise)
        obstacles = [Obstacle(coords, angle) for coords, angle in obstacle_list]
        
        prova = []
        # Draw the car
        car_art, car_patch, car_scatt = car.draw(ax)
    
        prova.append(car_art)
        prova.append(car_patch)
        prova.append(car_scatt)
        
        # Draw the car's corner
        corners = car.get_corners()
        for corner in corners:
            prova.append(ax.scatter(*corner, color="blue", zorder=5))
        
        # Draw the obstacles and relatives corners
        for obstacle in obstacles:
            obs_art, obs_patch, obs_scatt = obstacle.draw(ax)
            
            corners = obstacle.get_corners()
            for corner in corners:
                prova.append(ax.scatter(*corner, color="blue", zorder=5))
    
        return prova
    
        ani = animation.FuncAnimation(fig, animate, repeat=True,
                                            frames=len(states)-1, interval=50)
        
        # To save the animation using Pillow as a gif
        writer = animation.PillowWriter(fps=60,
                                        metadata=dict(artist='Me'),
                                        bitrate=1800)
        ani.save('scatter.gif', writer=writer)

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

def draw_trajectory_colored(states, start, obstacles, targets, target_list, has_completed, has_collided, save=False):    
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
    colors = ["blue" if completed.any() and not collided.any() else "red" for completed, collided in zip(has_completed, has_collided)]
    for state, color in zip(states, colors):
        ax.plot(state[:, 0], state[:, 1], color=color)
    
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

    plt.title(f"Completed: {has_completed.any(1).mean()}, Collided: {has_collided.any(1).mean()}")
    # Show the plot
    if save:
        plt.savefig("trajectories.png")
    plt.show()
    
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

def plot_inputs(inputs, references, save=False):
    ref_x, ref_y = references.transpose(2, 0, 1)
    
    plt.plot(np.arange(inputs.shape[1]), inputs.mean(0)[:, 0], label="x", zorder=3)
    plt.plot(np.arange(inputs.shape[1]), ref_x.mean(0), label="ref_x")
    plt.legend()
    plt.show()

    plt.plot(np.arange(inputs.shape[1]), inputs.mean(0)[:, 1], label="y", zorder=3)
    plt.plot(np.arange(inputs.shape[1]), ref_y.mean(0), label="ref_y")
    plt.legend()

    if save:
        plt.savefig("inputs.png")
    plt.show()
