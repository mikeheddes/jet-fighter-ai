import math
from collections import namedtuple
import random
import moderngl
import moderngl_window
from moderngl.context import Context
import numpy as np
from PIL import Image
import sys
from pyrr import Matrix44
import time
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from jet_fighter.plane import Plane, PlaneAction, PlaneState
from jet_fighter.bullet import BulletState

WIDTH = 240
HEIGHT = 180
# BLUE = '#0066ff'
BLUE = np.array((0.0, 0.4, 1.0), dtype="f4")
DEAD_BLUE = '#003366'
DEAD_BLUE = np.array((0.0, 0.2, 0.4), dtype="f4")
# RED = '#ff3333'
RED = np.array((1.0, 0.2, 0.2), dtype="f4")
DEAD_RED = '#660000'
DEAD_RED = np.array((0.4, 0.0, 0.0), dtype="f4")

FrameSize = namedtuple("FrameSize", ("width", "height"))


class JetFighter:
    def __init__(self, width, height):
        self.frameSize = FrameSize(width, height)

    def resetPlane(self, color, dead_color):
        x = random.random() * self.frameSize.width
        y = random.random() * self.frameSize.height
        angle = random.random() * math.pi * 2.0
        return Plane(x, y, angle, self.frameSize, color, dead_color)

    def reset(self):
        if hasattr(self, "planes"):
            for plane in self.planes:
                del plane

        self.score = [0, 0]
        self.planes = [self.resetPlane(
            BLUE, DEAD_BLUE), self.resetPlane(RED, DEAD_RED)]

    def step(self, actions):
        for index in range(len(self.planes)):
            plane = self.planes[index]
            plane.step(actions[index])

        for plane in self.planes:
            for bullet in plane.bullets:
                for index in range(len(self.planes)):
                    if self.planes[index].isIntersecting(bullet.x, bullet.y) and self.planes[index].state != PlaneState.DEAD:
                        self.planes[index].state = PlaneState.DEAD
                        bullet.state = BulletState.EXPLODING
                        # Make a new copy of the array so old != new
                        self.score = self.score.copy()
                        # Update score of the other player
                        self.score[(index + 1) % len(self.planes)] += 1

    def draw(self, ctx: moderngl.Context):
        for plane in self.planes:
            plane.draw(ctx)


class HeadlessRenderer:
    n_actions = 4
    state_size = (3, HEIGHT, WIDTH)
    samples = 8

    def __init__(self, maxlen=1500):
        if sys.platform.startswith('linux'):
            self.ctx = moderngl.create_context(standalone=True, backend='egl')
        else:
            self.ctx = moderngl.create_context(standalone=True)

        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)

        self.fbo = self.ctx.simple_framebuffer(
            (WIDTH, HEIGHT), samples=self.samples)
        self.fbo.use()

        self.fbo_out = self.ctx.simple_framebuffer((WIDTH, HEIGHT), samples=0)

        self.game = JetFighter(WIDTH, HEIGHT)
        self.maxlen = maxlen
        self.frame_num = 0

    def render(self):
        self.fbo.clear(0.0, 0.0, 0.0, 1)
        self.game.draw(self.ctx)

    def read_image(self):
        self.ctx.copy_framebuffer(dst=self.fbo_out, src=self.fbo)
        image = Image.frombuffer("RGB", self.fbo_out.size, self.fbo_out.read())
        return image

    def reset(self):
        self.frame_num = 0
        self.game.reset()
        self.last_score = self.game.score
        self.render()
        return self.read_image()

    def my_coordinates(self):
        return self.game.planes[0].x, self.game.planes[0].y

    def step(self, action):
        self.game.step([action, random.randint(0, 3)])

        score_change = [
            self.game.score[0] - self.last_score[0],
            self.game.score[1] - self.last_score[1],
        ]
        reward = score_change[0] - score_change[1]
        self.last_score = self.game.score
        self.render()
        self.frame_num += 1
        done = self.frame_num >= self.maxlen
        return self.read_image(), float(reward), done, {}


class App(moderngl_window.WindowConfig):
    window_size = (WIDTH // 2, HEIGHT // 2)
    resizable = False
    aspect_ratio = None
    clear_color = (0.0, 0.0, 0.0, 1.0)
    samples = 8

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)

        self.game = JetFighter(WIDTH, HEIGHT)
        self.game.reset()
        self.last_score = self.game.score
        # self.frame_num = 0

        self.action_repeat_counter = [0, 0]
        self.actions = [0, 0]

    def render(self, current_time, frame_time):
        for i in range(2):
            if self.action_repeat_counter[i] <= 0:
                self.action_repeat_counter[i] = random.randint(1, 120)
                self.actions[i] = random.randint(0, 3)

            self.action_repeat_counter[i] -= 1

        self.game.step(self.actions)
        score_change = [
            self.game.score[0] - self.last_score[0],
            self.game.score[1] - self.last_score[1],
        ]
        reward = score_change[0] - score_change[1]
        if reward != 0:
            print(reward)
            print(self.game.score)
        self.last_score = self.game.score
        self.game.draw(self.ctx)
        # time.sleep(0.1)
        # self.frame_num += 1

        # if self.frame_num == 240:
        #     image = Image.frombuffer("RGB", self.ctx.fbo.size, self.ctx.fbo.read())
        #     image.show()


if __name__ == "__main__":
    App.run()
