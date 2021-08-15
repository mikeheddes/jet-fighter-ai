import math
import moderngl
import numpy as np

from jet_fighter.utils import getInEdgeArea


class BulletState:
    FLYING = 1
    EXPLODING = 2
    DONE = 3


VERTEX_SHADER = '''
#version 330

uniform vec2 move;
uniform float explosion_growth;
uniform float isSecondAnimationStep;

void main() {
    gl_Position = vec4(move, 0.0, 1.0);
    gl_PointSize = 2.0 + explosion_growth * ((isSecondAnimationStep - 1.0) *  -1.0);
}
'''

FRAGMENT_SHADER = '''
#version 330

uniform float explosion_growth;
uniform float isSecondAnimationStep;

out vec3 f_color;

void main() {
    float r = length( gl_PointCoord - vec2( 0.5, 0.5 ) );
    if (r > .5) discard;
    if (isSecondAnimationStep == 0.0 && explosion_growth > 0.0 && r < 0.33) discard;
    f_color = vec3(0.8, 0.8, 0.8);
}
'''


class Bullet:
    velocity = 1.25
    lifetime = 140  # in frames
    framesPerExplosionStep = 30

    def __init__(self, x, y, angle, frameSize, color='#cccccc'):
        self.startX = x
        self.startY = y
        self.x = x
        self.y = y
        self.angle = angle
        self.color = color
        self.frameSize = frameSize
        self.lifetimeFrameCounter = 0
        self.state = BulletState.FLYING
        self.explosionFrameCounter = 0

        self.prog = None
        self.vao = None

    def __del__(self):
        if self.vao:
            self.vao.release()
            self.vao = None
        if self.prog:
            self.prog.release()
            self.prog = None

    def step(self):
        if self.state == BulletState.FLYING:
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

            self.lifetimeFrameCounter += 1

            if self.lifetimeFrameCounter >= self.lifetime:
                self.state = BulletState.EXPLODING
        elif self.state == BulletState.EXPLODING:
            self.explosionFrameCounter += 1

            if self.explosionFrameCounter == self.framesPerExplosionStep * 3:
                self.explosionFrameCounter = 0
                self.state = BulletState.DONE

    def draw(self, ctx: moderngl.Context):
        if self.state == BulletState.DONE:
            return

        if self.prog == None:
            self.prog = ctx.program(
                vertex_shader=VERTEX_SHADER,
                fragment_shader=FRAGMENT_SHADER)

        if self.vao == None:
            self.vao = ctx.vertex_array(self.prog, [])

        width = self.frameSize.width
        height = self.frameSize.height

        self.prog['explosion_growth'].write(np.array(
            2.0 if self.state == BulletState.EXPLODING else 0.0, dtype="f4"))
        self.prog['isSecondAnimationStep'].write(
            np.array(self.explosionFrameCounter % self.framesPerExplosionStep > self.framesPerExplosionStep / 2, dtype="f4"))


        def _draw(x, y):
            self.prog['move'].write(np.array((
                x / width * 2.0 - 1.0,
                y / height * 2.0 - 1.0), dtype='f4'))
    
            self.vao.render(moderngl.POINTS, vertices=1)

        _draw(self.x, self.y)
        inEdgeArea = getInEdgeArea(self.x, self.y, 1.5, self.frameSize)
        if inEdgeArea.top:
            _draw(self.x, self.y + height)
        if inEdgeArea.left:
            _draw(self.x + width, self.y)
        if inEdgeArea.right:
            _draw(self.x - width, self.y)
        if inEdgeArea.bottom:
            _draw(self.x, self.y - height)
