import gym 
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

import math
"""
Position beams in front of the car - more realistic and less work without 
removing a car.
* Change the values of the outside of the road to some < 255
* Draw a line in a color that will increase the value to 255 
* Find the 255 point - that is the intersection 
* Calculate distance between beam start and that intersection
There could be a situation when using this method, when a line will be drawn
with some weird angle and in that case more than one pixel can be tagged. Then
just pick one randomly to save time (they will be basically at the same spot).
Or maybe find some way of determining which one is closer (considering middle
of the road and which has closer index to that or smth simple not to slow it
down)

Consider what contours to draw:
    * all with -1? what if some false positives?
    * only the biggest with sorting? 

"""

# Constants 
FRAMES_2_SKIP = 100

SCREEN_WIDTH = SCREEN_HEIGHT = 300#95

# Car position 
CAR_X = 47
CAR_Y = 66

# Ranges get the road into binary image
LOWERB = np.array([0, 0 ,0])
UPPERB = np.array([179, 120, 230])


def resize(rgb_img):
    global SCREEN_WIDTH, SCREEN_HEIGHT
    fy = 1.0 / (rgb_img.shape[0] / SCREEN_HEIGHT)
    fx = 1.0 / (rgb_img.shape[1] / SCREEN_WIDTH)
    return cv.resize(rgb_img, (0, 0), fx=fx, fy=fy)


def get_bin_road(img):
    global LOWERB, UPPERB 
    return cv.inRange(img, LOWERB, UPPERB)


class Ray:
    def __init__(self, x, y, angle_deg, img_shape: tuple):
        # Calculate start and end coords for the line 
        self.start_x = x  
        self.start_y = y
        self.end_x = int(round(1000 * math.cos(math.pi * angle_deg / 180.0))) + self.start_x
        self.end_y = int(round(1000 * math.sin(math.pi * angle_deg / 180.0))) + self.start_y
        print("X: ", self.end_x, " Y: ", self.end_y)
        self.angle = angle_deg
        self.ray_matrice = np.zeros(img_shape)
        self.casted = None

        # Draw a line onto the ray_matrice
        cv.line(self.ray_matrice, # Inverted x/y?
                (self.start_x, self.start_y),
                (self.end_x, self.end_y), 125)
    
    # TODO: make values of pixels as constants
    def cast(self, cont_img):
        self.casted = cv.add(cont_img, self.ray_matrice)
        #plt.imshow(self.casted)
        #plt.show()
    
    def get_intersection(self):
        try:
            out = np.argwhere(self.casted == 250)[0]
            return (out[1], out[0])
        except IndexError:
            print("No result of intersection for {}".format(self.angle))
            return

    #def


if __name__ == '__main__':
    env = gym.make("CarRacing-v1")
    env.reset()
    
    rays = {
            'ray0' : Ray(CAR_X, CAR_Y, 0, (SCREEN_HEIGHT, SCREEN_WIDTH)),
            'ray_p180' : Ray(CAR_X, CAR_Y, 180, (SCREEN_HEIGHT, SCREEN_WIDTH)),
            'ray_p45' : Ray(CAR_X, CAR_Y, 45, (SCREEN_HEIGHT, SCREEN_WIDTH)),
            'ray_m45' : Ray(CAR_X, CAR_Y, -45, (SCREEN_HEIGHT, SCREEN_WIDTH)),
            'ray_p135' : Ray(CAR_X, CAR_Y, 135, (SCREEN_HEIGHT, SCREEN_WIDTH)),
            'ray_m135' : Ray(CAR_X, CAR_Y, -135, (SCREEN_HEIGHT, SCREEN_WIDTH)),
            'ray_m90' : Ray(CAR_X, CAR_Y, -90, (SCREEN_HEIGHT, SCREEN_WIDTH))
            }

    for i in range(1000):
        if i > FRAMES_2_SKIP:
            # RGB frame 
            screen_original = env.render(mode="rgb_array")

            # Reduce the img from the bar at the bottom
            screen = screen_original[0:300, :]
            
            # Resized frame
            screen_96_96 = resize(screen)
            
            # As binary image containing only road
            binary_screen = get_bin_road(screen_96_96)
            contours, _ = cv.findContours(binary_screen, cv.RETR_EXTERNAL, 
                    cv.CHAIN_APPROX_SIMPLE) 
            
            # Draw only the contour
            road_cont = np.zeros(shape=screen_96_96.shape[:2])
            cv.drawContours(road_cont, contours, -1, 125, 1) # TODO change 125 to constant

            # Cast rays
            temp = road_cont.copy()
            for ray in rays.values():
                ray.cast(road_cont)
                intersection = ray.get_intersection()
                print(intersection)
                cv.circle(temp, intersection, 5, 150, -1)
            
            cv.imshow('asdf', temp)
            cv.waitKey(5)
            """
            fig, axs = plt.subplots(4, 1)
            fig.suptitle('Processing')
            
            axs[0].imshow(screen_original)
            axs[0].set_title('Original')
            
            axs[1].imshow(binary_screen)
            axs[1].set_title('Road binary')
            
            axs[2].imshow(road_cont)
            axs[2].set_title('Road contour')
            
            axs[3].imshow(temp)
            axs[3].set_title('Intersection')
            plt.show()
            """

        env.step(env.action_space.sample())

    env.close()








