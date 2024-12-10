import math
import numpy as np

from PIL import Image                                        # To Rotate the images

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox # To Resize and add the images
import matplotlib.patches as patches
from matplotlib.image import imread                          # To open the images

class Obstacle():
    def __init__(self, coords, theta):
        self.x     = coords[0]
        self.y     = coords[1]

        self.theta = theta
        self.img   = Image.fromarray((imread("images/barrier.png") * 255).astype(np.uint8))

        self.height = 15
        self.width  = 30        

        self.corners = np.array([ (+ self.width, + self.height), 
                                  (- self.width, + self.height),
                                  (- self.width, - self.height),
                                  (+ self.width, - self.height), ])

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
        # Get the Obstacle's coordinates
        return np.array([self.x, self.y])
    
    def draw(self, ax):
        # Retrieve the rotated image by the car's angle
        obstacle_rotated = self.img.rotate(self.theta)
        obstacle_offset = OffsetImage(obstacle_rotated, zoom=0.05)
        obstacle_annotation = AnnotationBbox(obstacle_offset, (self.x, self.y), frameon=False)
        
        # Add the car and the hitbox to the plot
        artist = ax.add_artist(obstacle_annotation)
        patch  = ax.add_patch( patches.Rectangle((self.x - self.width, self.y - self.height), 
                                        height=self.height*2, width=self.width*2, angle=self.theta, 
                                        rotation_point="center", 
                                        edgecolor='red', facecolor="None", lw=3, 
                                        zorder=5))
        
        # Add the center to the plot
        scatt  = ax.scatter(self.x, self.y, color="red", zorder=5)

        return artist, patch, scatt