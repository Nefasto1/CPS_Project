import math
import numpy as np

from PIL import Image                                        # To Rotate the images

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox # To Resize and add the images
import matplotlib.patches as patches
from matplotlib.image import imread                          # To open the images


class Car():
    def __init__(self, coords, speed, theta=None, noise=[1, 1, 0.5, 0.5]):
        self.x = coords[0]
        self.y = coords[1]
        
        self.speed_x = speed[0]
        self.speed_y = speed[1]
        
        # Determine the angle of the car, the image is rotated by default of -90 degrees
        self.theta   = (theta + 90) % 360 if theta is not None else (np.rad2deg(np.arctan2(self.speed_y, self.speed_x)) + 90) % 360
        self.img     = Image.fromarray((imread("images/car.png") * 255).astype(np.uint8))

        self.noise = noise
        
        self.height = 42
        self.width  = 22

        self.dt      = 0.5
        self.inertia = 0.8

        self.corners = np.array([ (+ self.width, + self.height), 
                                  (- self.width, + self.height),
                                  (- self.width, - self.height),
                                  (+ self.width, - self.height), ])    
        
    def move(self):
        # Move forward in the direction of the speeds
        self.x += self.speed_x * self.dt
        self.y += self.speed_y * self.dt

        return self.speed_x, self.speed_y
        
    def move_noisy(self):
        # Move forward in the direction of the speeds adding some normal noise
        self.x += self.speed_x * self.dt + np.random.normal(0, self.noise[0])
        self.y += self.speed_y * self.dt + np.random.normal(0, self.noise[1])

        return self.speed_x, self.speed_y

    def accelerate(self, acceleration):
        # Increase the speed by the acceleration in input
        self.speed_x = self.inertia * self.speed_x + acceleration[0] * self.dt
        self.speed_y = self.inertia * self.speed_y + acceleration[1] * self.dt

        # Determine the angle of the car, the image is rotated by default of -90 degrees
        self.theta = (np.rad2deg(np.arctan2(self.speed_y, self.speed_x)) + 90)%360
        
        return self.move()
        
    def accelerate_noisy(self, acceleration):
        # Increase the speed by the acceleration in input plus a normal noise
        self.speed_x = self.inertia * self.speed_x + acceleration[0] * self.dt + np.random.normal(0, self.noise[2])
        self.speed_y = self.inertia * self.speed_y + acceleration[1] * self.dt + np.random.normal(0, self.noise[3])
            
        # Determine the angle of the car, the image is rotated by default of -90 degrees
        self.theta = (np.rad2deg(np.arctan2(self.speed_y, self.speed_x)) + 90)%360
        
        return self.move_noisy()

    def rotate_corners(self):
        # Rotate the corners based on the car's angle
        x = self.corners[:, 0] * math.cos(math.radians(self.theta)) \
          - self.corners[:, 1] * math.sin(math.radians(self.theta))

        y = self.corners[:, 0] * math.sin(math.radians(self.theta)) \
          + self.corners[:, 1] * math.cos(math.radians(self.theta))

        return np.hstack((x[:, np.newaxis], y[:, np.newaxis]))

    def get_corners(self):
        # Retrieve the rotated corners
        corners = self.rotate_corners()

        # Shift the corners to the car's position
        corners[:, 0] = self.x + corners[:, 0]
        corners[:, 1] = self.y + corners[:, 1]
        
        return corners

    def get_coords(self):
        # Get the coordinates
        return np.array([self.x, self.y])
        
    def get_state(self):
        # Get the state
        return np.array([self.x, self.y, self.speed_x, self.speed_y])

    def get_state_noisy(self):
        # Get the state plus a normal noise
        return self.get_state() + np.random.normal(0, self.noise, (4, ))
        
    def draw(self, ax):
        # Retrieve the rotated image by the car's angle
        car_rotated = self.img.rotate(self.theta)
        car_offset = OffsetImage(car_rotated, zoom=0.1)
        car_annotation = AnnotationBbox(car_offset, (self.x, self.y), frameon=False)

        # Add the car and the hitbox to the plot
        ax.add_artist(car_annotation)
        ax.add_patch( patches.Rectangle((self.x - self.width, self.y - self.height), 
                                        height=self.height*2, width=self.width*2, angle=self.theta, 
                                        rotation_point="center", 
                                        edgecolor='red', facecolor="None", lw=3, 
                                        zorder=5))
        
        # Add the center to the plot
        plt.scatter(self.x, self.y, color="red", zorder=5)