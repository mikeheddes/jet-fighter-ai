from torch.utils.tensorboard import SummaryWriter
import datetime
import os

BATCH_SIZE = 128
GAMMA = 0.99
EPS_DECAY = 20_000
LEARNING_RATE = 0.0005
C, H, W = 1, 90, 120
NUM_ACTIONS = 2
STACKING = 4
MEMORY_SIZE = 100_000
TARGET_NET_UPDATE_FREQ = 500
NUM_STEPS = 150_000

os.makedirs("../runs/", exist_ok=True)


class VariableHandler:
    def __init__(self):
        self.step = 0

    def init_writer(self, name):
        timezone = datetime.timezone.utc
        current_date = datetime.datetime.now(timezone)
        version = current_date.strftime("d%Y_%m_%d-t%H_%M_%S")
        self.writer = SummaryWriter(os.path.join("../runs", name, version))

    def get_step(self):
        return self.step

    def set_step(self, step):
        self.step = step


variables = VariableHandler()