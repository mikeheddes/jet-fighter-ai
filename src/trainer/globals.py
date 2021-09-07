from torch.utils.tensorboard import SummaryWriter
import datetime
import os

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 20_000
LEARNING_RATE = 0.0005
C, H, W = 1, 1, 4
NUM_ACTIONS = 2
STACKING = 4
MEMORY_SIZE = 100_000
TARGET_NET_UPDATE_FREQ = 500
NUM_STEPS = 150_000

os.makedirs("../runs/", exist_ok=True)

timezone = datetime.timezone.utc
current_date = datetime.datetime.now(timezone)
version = current_date.strftime("d%Y_%m_%d-t%H_%M_%S")
writer = SummaryWriter(f'../runs/dqn_cartpole/{version}/')

class VariableHandler:
    def __init__(self):
        self.step = 0

    def get_step(self):
        return self.step

    def set_step(self, step):
        self.step = step


variables = VariableHandler()
