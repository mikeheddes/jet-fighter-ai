from torch.utils.tensorboard import SummaryWriter
import datetime
import os
import ray

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 20_000
LEARNING_RATE = 0.001
C, H, W = 1, 1, 4
NUM_ACTIONS = 2
STACKING = 1
MEMORY_SIZE = 100_000
TARGET_NET_UPDATE_FREQ = 500
NUM_STEPS = 150_000


@ray.remote
class Commons:
    def __init__(self):
        self.step = 0

        os.makedirs("../runs/", exist_ok=True)
        timezone = datetime.timezone.utc
        current_date = datetime.datetime.now(timezone)
        version = current_date.strftime("d%Y_%m_%d-t%H_%M_%S")
        self.writer = SummaryWriter(f'../runs/dqn_cartpole/{version}/')

    def get_step(self):
        return self.step

    def set_step(self, step):
        self.step = step

    def write_scalar(self, *args, **kwargs):
        self.writer.add_scalar(*args, **kwargs, global_step=self.step)

    def write_histogram(self, *args, **kwargs):
        self.writer.add_histogram(*args, **kwargs, global_step=self.step)

    def write_video(self, *args, **kwargs):
        self.writer.add_video(*args, **kwargs, global_step=self.step)

    def write_graph(self, *args, **kwargs):
        self.writer.add_graph(*args, **kwargs)

