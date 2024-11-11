from src.utils import rectangles_overlap

import math
import numpy as np

from PIL import Image                                        # To Rotate the images

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox # To Resize and add the images
import matplotlib.patches as patches
from matplotlib.image import imread                          # To open the images


class Car():
    def __init__(self, coords, speed, theta=None):
        self.x     = coords[0]
        self.y     = coords[1]
        self.speed_x = speed[0]
        self.speed_y = speed[1]
        self.theta = (theta + 90) % 360 if theta is not None else (np.rad2deg(np.arctan2(self.speed_y, self.speed_x)) + 90) % 360
        self.img   = Image.fromarray((imread("images/car.png") * 255).astype(np.uint8))
        
        self.height = 42
        self.width  = 22

        self.dt      = 0.5
        self.inertia = 0.8

        self.corners = np.array([ (+ self.width, + self.height), 
                                  (- self.width, + self.height),
                                  (- self.width, - self.height),
                                  (+ self.width, - self.height), ])
            
        
    def move(self):
        # Move forward in the direction of the angle
        self.x += self.speed_x * self.dt
        self.y += self.speed_y * self.dt

        return self.speed_x, self.speed_y
        
    def move_noisy(self):
        # Move forward in the direction of the angle
        self.x += self.speed_x * self.dt + np.random.normal(0, 1)
        self.y += self.speed_y * self.dt + np.random.normal(0, 1)

        return self.speed_x, self.speed_y

    def accelerate(self, accelleration):
        self.speed_x = self.inertia * self.speed_x + accelleration[0] * self.dt
        self.speed_y = self.inertia * self.speed_y + accelleration[1] * self.dt

        self.theta = (np.rad2deg(np.arctan2(self.speed_y, self.speed_x)) + 90)%360
        
        return self.move()
        
    def accelerate_noisy(self, accelleration):
        self.speed_x = self.inertia * self.speed_x + accelleration[0] * self.dt + np.random.normal(0, 0.5)
        self.speed_y = self.inertia * self.speed_y + accelleration[1] * self.dt + np.random.normal(0, 0.5)

        self.theta = (np.rad2deg(np.arctan2(self.speed_y, self.speed_x)) + 90)%360
        
        return self.move_noisy()
        
    def draw(self, ax):
        car_rotated = self.img.rotate(self.theta)
        car_offset = OffsetImage(car_rotated, zoom=0.1)
        car_annotation = AnnotationBbox(car_offset, (self.x, self.y), frameon=False)
        
        ax.add_artist(car_annotation)
        ax.add_patch( patches.Rectangle((self.x - self.width, self.y - self.height), 
                                        height=self.height*2, width=self.width*2, angle=self.theta, 
                                        rotation_point="center", 
                                        edgecolor='red', facecolor="None", lw=3, 
                                        zorder=5))
        plt.scatter(self.x, self.y, color="red", zorder=5)

    def rotate_corners(self):
        x = self.corners[:, 0] * math.cos(math.radians(self.theta)) \
          - self.corners[:, 1] * math.sin(math.radians(self.theta))

        y = self.corners[:, 0] * math.sin(math.radians(self.theta)) \
          + self.corners[:, 1] * math.cos(math.radians(self.theta))

        return np.hstack((x[:, np.newaxis], y[:, np.newaxis]))

    def get_corners(self):
        # Rotate the corners
        corners = self.rotate_corners()

        corners[:, 0] = self.x + corners[:, 0]
        corners[:, 1] = self.y + corners[:, 1]
        
        return corners

    def is_overlapping(self, obstacles):
        for obstacle in obstacles:
            if rectangles_overlaps(car, obstacle):
                return True
        return False

    def get_state(self):
        return np.array([self.x, self.y, self.speed_x, self.speed_y])

    def get_state_noisy(self):
        return self.get_state() + np.random.normal(0, 1, (4, ))
