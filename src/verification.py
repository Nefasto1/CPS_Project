import matplotlib.pyplot as plt
from src.utils import get_modules
import numpy as np

def plot_quantitative_boolean(y, boolean, quantitative, last, name=""):
    plt.figure(figsize=(15, 10))
    
    plt.plot(np.arange(last), y[:last], lw=3, label="modules")
    plt.plot(np.arange(last), boolean[:last], lw=1, label="boolean")
    plt.plot(np.arange(min(last, len(quantitative))), quantitative[:last], lw=1, label="quantitative")
    
    plt.axvline(last, color="red")
    plt.axhline(0, color="black")
    if name == "Collision":
        plt.title(f"{name}: {boolean.any()}")
    else:
        plt.title(f"{name} Proportion: {round(np.mean(boolean), 3)}")
    plt.legend()
    plt.show()

def safe_turns(modules, theta_list):
    thetas = np.rad2deg(theta_list)            # Convert radiants to degree
    thetas = np.abs(thetas[:-1] - thetas[1:])  # Take the difference to the previous timestep

    turns     = np.vstack((thetas - 45, modules - 1.5))
    turns[0] /= 360                                  # Maximum angle
    turns[1] /= 5                                    # Maximum Speed
    
    # Evaluate if: (Theta > 45) => (speed <= 1.5)
    boolean      = (thetas <= 45) | (modules <= 1.5)
    quantitative = np.min(turns, axis=0) * 5         # Minimum for the or condition

    return boolean, quantitative

def safe_speed(states, threshold=5):
    boolean      = get_modules(states[:, 2:]) < threshold
    quantitative = get_modules(states[:, 2:]) - threshold
    
    return boolean, quantitative

def collision(has_collided):
    boolean = has_collided
    last    = boolean.argmax()

    return boolean, last

def verification(states, theta_list, has_collided, has_completed, speed_threshold=5):
    modules = get_modules(states[1:, 2:])      # Take the speeds at each timestep
    last    = has_completed.argmax()
    
    boolean_turn, quantitative_turn   = safe_turns(modules, theta_list)
    boolean_speed, quantitative_speed = safe_speed(states, speed_threshold)
    

    plot_quantitative_boolean(modules, boolean_turn, quantitative_turn, last, "Safe Turns")
    plot_quantitative_boolean(modules, boolean_speed, quantitative_speed, last, "Safe Speed")
    plot_quantitative_boolean(modules, has_collided, [], last, "Collision")