import gym 
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

import math

import torch 
from torch.nn.functional import normalize
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
When car goes out of the track, it will stop detecting intersections - stop the 
training then.

"""

# Constants 
FRAMES_2_SKIP = 100

SCREEN_WIDTH = SCREEN_HEIGHT = 300

# Car position 
CAR_X = int(SCREEN_WIDTH * 0.5)
CAR_Y = int(SCREEN_HEIGHT * 0.9) # 0.6875

# Ranges get the road into binary image
LOWERB = np.array([0, 0 ,0])
#UPPERB = np.array([179, 120, 230])
UPPERB = np.array([179, 255, 122])


def resize(rgb_img):
    global SCREEN_WIDTH, SCREEN_HEIGHT
    fy = 1.0 / (rgb_img.shape[0] / SCREEN_HEIGHT)
    fx = 1.0 / (rgb_img.shape[1] / SCREEN_WIDTH)
    return cv.resize(rgb_img, (0, 0), fx=fx, fy=fy)


def get_bin_road(img):
    global LOWERB, UPPERB 
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    return cv.inRange(hsv_img, LOWERB, UPPERB)


class Ray:
    def __init__(self, x, y, angle_deg, img_shape: tuple):
        # Calculate start and end coords for the line 
        self.start_x = x  
        self.start_y = y
        self.end_x = None
        self.end_y = None
        init_end_x = int(round(1000 * math.cos(math.pi * angle_deg / 180.0))) + self.start_x
        init_end_y = int(round(1000 * math.sin(math.pi * angle_deg / 180.0))) + self.start_y
        self.angle = angle_deg
        self.ray_matrice = np.zeros(img_shape)
        self.casted = None
        self.distance = -1

        # Draw a line onto the ray_matrice
        cv.line(self.ray_matrice, # Inverted x/y?
                (self.start_x, self.start_y),
                (init_end_x, init_end_y), 125)
    
    # TODO: make values of pixels as constants
    def cast(self, cont_img):
        self.casted = cv.add(cont_img, self.ray_matrice)
        #plt.imshow(self.casted)
        #plt.show()
    
    def calculate_intersection(self):
        try:
            out = np.argwhere(self.casted == 250)
            if len(out) == 1:
                self.end_x = out[0, 1]
                self.end_y = out[0, 0]
                self.distance = math.sqrt(
                            math.pow(self.start_x - out[0, 1], 2) + 
                            math.pow(self.start_y - out[0, 0], 2))

            else:
                smallest_distance = 1000
                smallest_d_idx = -1
                for idx, pt in enumerate(out):
                    dist = math.sqrt(
                            math.pow(self.start_x - pt[1], 2) + 
                            math.pow(self.start_y - pt[0], 2))
                    if dist < smallest_distance:
                        smallest_distance = dist
                        smallest_d_idx = idx
                self.end_x = out[smallest_d_idx, 1]
                self.end_y = out[smallest_d_idx, 0]
                self.distance = smallest_distance

        except IndexError:
            print("No result of intersection for {}".format(self.angle))
            return 

    def get_distance_int(self):
        return self.distance

    def get_intersection(self):
        return (self.end_x, self.end_y)


def get_state_as_list(env, rays: list):
    # Get and resize the original image
    screen_original = env.render(mode="rgb_array")
    screen = screen_original[0:300, :]
    screen_96_96 = resize(screen)

    # Convert to binary and get contours, draw them
    binary_screen = get_bin_road(screen_96_96)
    contours, _ = cv.findContours(binary_screen, cv.RETR_EXTERNAL, 
            cv.CHAIN_APPROX_SIMPLE) 
    road_cont = np.zeros(shape=screen_96_96.shape[:2])
    cv.drawContours(road_cont, contours, -1, 125, 5) # TODO change 125 to constant
    
    # Cast rays
    state = []
    for ray in rays:
        ray.cast(road_cont)
        ray.calculate_intersection()
        distance = ray.get_distance_int()
        state.append(distance)
    return state 


def get_state(env, rays: list):
    state_list = get_state_as_list(env, rays)
    state_np = np.reshape(np.array(state_list), (1, -1))
    tensor = normalize(torch.from_numpy(state_np))
    return tensor
        


if __name__ == '__main__':
    env = gym.make("CarRacing-v1")
    env.reset()
    
    """
    rays = {
            'ray0' : Ray(CAR_X, CAR_Y, 0, (SCREEN_HEIGHT, SCREEN_WIDTH)),
            'ray_p180' : Ray(CAR_X, CAR_Y, 180, (SCREEN_HEIGHT, SCREEN_WIDTH)),
            'ray_m45' : Ray(CAR_X, CAR_Y, -45, (SCREEN_HEIGHT, SCREEN_WIDTH)),
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
            cv.drawContours(road_cont, contours, -1, 125, 5) # TODO change 125 to constant

            # Cast rays
            temp = road_cont.copy()
            print('##############################')
            for ray in rays.values():
                ray.cast(road_cont)
                ray.calculate_intersection()
                distance = ray.get_distance_int()
                print('For ray with angle: {}, distance is equal to: {}'.format(ray.angle, distance))
                intersection = ray.get_intersection()
                cv.line(temp, (ray.start_x, ray.start_y), intersection, 50, 1)
            
            cv.imshow('asdf', temp)
            cv.waitKey(5)
    """
    rays = [
            Ray(CAR_X, CAR_Y, 0, (SCREEN_HEIGHT, SCREEN_WIDTH)),
            Ray(CAR_X, CAR_Y, 180, (SCREEN_HEIGHT, SCREEN_WIDTH)),
            Ray(CAR_X, CAR_Y, -45, (SCREEN_HEIGHT, SCREEN_WIDTH)),
            Ray(CAR_X, CAR_Y, -135, (SCREEN_HEIGHT, SCREEN_WIDTH)),
            Ray(CAR_X, CAR_Y, -90, (SCREEN_HEIGHT, SCREEN_WIDTH))
            ]

    for i in range(1000):
        if i > FRAMES_2_SKIP:
            state = get_state_as_list(env, rays)
            temp = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH))
            for ray in rays:
                cv.line(temp, (ray.start_x, ray.start_y), (ray.end_x, ray.end_y), 150)
            cv.imshow('asdf', temp)
            cv.waitKey(1)
        
        env.step(env.action_space.sample())

    env.close()








