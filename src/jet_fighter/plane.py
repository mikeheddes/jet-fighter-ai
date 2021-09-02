import math
import moderngl
import numpy as np
from pyrr import Matrix44

from jet_fighter.bullet import Bullet, BulletState
from jet_fighter.utils import getInEdgeArea


class PlaneAction:
    ROTATE_LEFT = 0
    ROTATE_RIGHT = 1
    FIRE = 2
    NOTHING = 3


class PlaneState:
    ALIVE = 0
    DEAD = 1


VERTEX_SHADER = '''
#version 330

uniform vec2 move;
uniform vec3 color;
uniform mat4 rot;
uniform vec2 stretch;
in vec2 in_vert;

out vec3 v_color;

void main() {
    v_color = color;

    vec4 pos = vec4(in_vert, 0.0, 1.0);
    pos = rot * pos;
    pos.x *= stretch.x;
    pos += vec4(move, 0.0, 0.0);
    gl_Position = pos;
}
'''
FRAGMENT_SHADER = '''
#version 330

in vec3 v_color;

out vec3 f_color;

void main() {
    f_color = v_color;
}
'''


class Plane:
    borderBox = 3.5
    velocity = 0.75
    turnSpeed = 0.025
    deadStepsPerAnimationLoop = 40
    fireLimit = 30  # steps

    def __init__(self, x, y, angle, frameSize, color, dead_color):
        self.x = x
        self.y = y
        self.angle = angle
        self.color = color
        self.dead_color = dead_color
        self.frameSize = frameSize
        self.state = PlaneState.ALIVE
        self.bullets = []
        self.deadStepCounter = 0
        self.canFireStepCountDown = 0

        self.prog = None
        self.vao = None

    def __del__(self):
        if hasattr(self, "bullets"):
            while len(self.bullets) > 0:
                bullet = self.bullets.pop()
                del bullet
            self.bullets = None
        if self.buffer:
            self.buffer.release()
            self.buffer = None
        if self.vao:
            self.vao.release()
            self.vao = None
        if self.prog:
            self.prog.release()
            self.prog = None

    def step(self, action):
        # Handle fire rate limit
        if action == PlaneAction.FIRE and self.canFireStepCountDown != 0:
            action = PlaneAction.NOTHING

        if self.canFireStepCountDown != 0:
            self.canFireStepCountDown -= 1

        if action == PlaneAction.FIRE:
            # Reset firing wait time
            self.canFireStepCountDown = self.fireLimit

            x = self.x + self.borderBox * math.cos(self.angle)
            y = self.y + self.borderBox * math.sin(self.angle)
            self.bullets.append(Bullet(x, y, self.angle, self.frameSize))

        for index in range(len(self.bullets) - 1, -1, -1):
            self.bullets[index].step()
            # Remove bullets that have already exploded.
            if self.bullets[index].state == BulletState.DONE:
                bullet = self.bullets.pop(index)
                del bullet

        # Update dead state
        # reset state if max dead frames is reached
        if self.state == PlaneState.DEAD:
            self.deadStepCounter += 1
            if self.deadStepCounter >= self.deadStepsPerAnimationLoop * 3:
                self.deadStepCounter = 0
                self.state = PlaneState.ALIVE

        if action == PlaneAction.NOTHING:
            pass

        if action == PlaneAction.ROTATE_LEFT:
            self.angle -= self.turnSpeed
        elif action == PlaneAction.ROTATE_RIGHT:
            self.angle += self.turnSpeed

        # Update position
        self.x += self.velocity * math.cos(self.angle)
        self.y += self.velocity * math.sin(self.angle)

        if self.x >= self.frameSize.width:
            self.x -= self.frameSize.width
        elif self.x < 0:
            self.x += self.frameSize.width

        if self.y >= self.frameSize.height:
            self.y -= self.frameSize.height
        elif self.y < 0:
            self.y += self.frameSize.height

    def draw(self, ctx: moderngl.Context):
        for bullet in self.bullets:
            bullet.draw(ctx)

        width = self.frameSize.width
        height = self.frameSize.height

        if self.prog == None:
            self.prog = ctx.program(
                vertex_shader=VERTEX_SHADER,
                fragment_shader=FRAGMENT_SHADER)

        if self.vao == None:
            vertices = np.array((
                3.5, 0,
                -3, 2.5,
                -2, 0,
                3.5, 0,
                -3, -2.5,
                -2, 0,
            ), dtype="f4") / height * 2.0
            self.buffer = ctx.buffer(vertices)
            self.vao = ctx.simple_vertex_array(
                self.prog, self.buffer, "in_vert")
            self.prog['stretch'].write(
                np.array((height / width, 1.0), dtype='f4'))

        self.prog['color'].write(
            self.color if self.state != PlaneState.DEAD else self.dead_color)
        self.prog['rot'].write(Matrix44.from_eulers(
            (0.0, -self.angle, 0.0), dtype='f4'))

        def _draw(x, y):
            self.prog['move'].write(np.array((
                x / width * 2.0 - 1.0,
                y / height * 2.0 - 1.0), dtype='f4'))

            self.vao.render(moderngl.TRIANGLES)

        _draw(self.x, self.y)
        inEdgeArea = getInEdgeArea(
            self.x, self.y, self.borderBox, self.frameSize)
        if inEdgeArea.top:
            _draw(self.x, self.y + height)
        if inEdgeArea.left:
            _draw(self.x + width, self.y)
        if inEdgeArea.right:
            _draw(self.x - width, self.y)
        if inEdgeArea.bottom:
            _draw(self.x, self.y - height)

    def isIntersecting(self, x, y):
        diff_x = self.x - x
        diff_y = self.y - y
        dist = math.sqrt(diff_x ** 2 + diff_y ** 2)
        return dist < self.borderBox
