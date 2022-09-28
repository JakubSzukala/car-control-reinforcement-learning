import gym
from stable_baselines3.common.env_checker import check_env
from distance_extractor import Ray
from distance_extractor import get_state
from distance_extractor import get_bin_road
import cv2 as cv
import numpy as np
from gym import spaces

class CarRacingDistanceStateWrapper(gym.Env):
    def __init__(self, car_racing_env):
        super(CarRacingDistanceStateWrapper, self).__init__()
        self.env = car_racing_env
        self.action_space = self.env.action_space
        # Ugly ugly hard coding, barf
        low = np.array([0, 0, 0, 0, 0], dtype=np.int16)
        high = np.array([150, 150, 150, 150, 150], dtype=np.int16)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int16) # type: ignore
        img_shape = list(car_racing_env.observation_space.sample().shape)
        img_shape[0] += 4
        img_shape[1] += 4
        car_x = int(img_shape[1] * 0.5)
        car_y = int(img_shape[0] * 0.6)

        self.rays = [
            Ray(car_x, car_y, 0, tuple(img_shape[:2])),
            Ray(car_x, car_y, 180, tuple(img_shape[:2])),
            Ray(car_x, car_y, -45, tuple(img_shape[:2])),
            Ray(car_x, car_y, -135, tuple(img_shape[:2])),
            Ray(car_x, car_y, -90, tuple(img_shape[:2]))
            ]


    def step(self, action):
        state, reward, done, info = self.env.step(action)
        
        # Get binary image
        bin_road = get_bin_road(state)
        
        # Debug
        temp = np.zeros(state.shape[:2])
        for ray in self.rays:
            cv.line(temp, (ray.start_x, ray.start_y), (ray.end_x, ray.end_y), 150)
                
        # Apply road contours to the image 
        contours, _ = cv.findContours(bin_road, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        road_cont_img = np.zeros(state.shape[:2])
        cv.drawContours(road_cont_img, contours, -1, 125, 2) 
        road_cont_img = cv.copyMakeBorder(road_cont_img, 2, 2, 2, 2, cv.BORDER_CONSTANT, value=125)
        road_cont_img = road_cont_img.astype(np.uint8)

        cv.imshow('road_cont_img', road_cont_img)
        cv.imshow('asdf', temp)
        cv.waitKey(1)

        state_as_distance = []
        for ray in self.rays:
            ray.cast(road_cont_img)
            ray.calculate_intersection()
            state_as_distance.append(ray.get_distance_int())
        new_state = np.array(state_as_distance, dtype=np.int16)
        return new_state, reward, done, info

    def reset(self):
        self.env.reset()
        return self.step(None)[0]

    def render(self, mode='human'):
        self.env.render()

if __name__ == '__main__':
    root_env = gym.make('CarRacing-v2', continuous=False)
    print(root_env.observation_space.sample().shape)
    env = CarRacingDistanceStateWrapper(root_env)
    check_env(env)
