import gym 
import cv2 as cv
from matplotlib import pyplot as plt

# Constants 
FRAMES_2_SKIP = 100

SCREEN_WIDTH = SCREEN_HEIGHT = 95.0

def resize(rgb_img):
    global SCREEN_WIDTH, SCREEN_HEIGHT
    fy = 1.0 / (rgb_img.shape[0] / SCREEN_HEIGHT)
    fx = 1.0 / (rgb_img.shape[1] / SCREEN_WIDTH)
    return cv.resize(rgb_img, (0, 0), fx=fx, fy=fy)

if __name__ == '__main__':
    env = gym.make("CarRacing-v1")
    env.reset()
    
    for i in range(1000):
        if i > FRAMES_2_SKIP:
            screen = env.render(mode="rgb_array")
            screen_95_95 = resize(screen)
            cv.circle(screen_95_95, (47, 66), 1, (0, 0, 150), 1)
            plt.imshow(screen_95_95)
            plt.show()

        env.step(env.action_space.sample())

    env.close()

