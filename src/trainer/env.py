import random
import gym
import numpy as np
from gym.spaces import Discrete, Box
from jet_fighter.game import HEIGHT, WIDTH, HeadlessRenderer


def toTensor(image):
    t = np.array(image, dtype=np.float32)
    t /= 255.0
    return t


class JetFighter(gym.Env):
    def __init__(self, config):
        self.renderer = HeadlessRenderer()

        self.action_space = Discrete(self.renderer.n_actions)
        self.observation_space = Box(
            0.0, 1.0, shape=(HEIGHT, WIDTH, 3), dtype=np.float32)

    def reset(self):
        image = self.renderer.reset()
        return toTensor(image)

    def step(self, action):
        image, reward, done, info = self.renderer.step(action)
        return toTensor(image), reward, done, info

    def seed(self, seed=None):
        random.seed(seed)
