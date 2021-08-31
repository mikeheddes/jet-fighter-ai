import random
from collections import UserList


class UniformReplay(UserList):
    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity
        self.wrap_arounds = 0
        self.write_index = 0

    def append(self, item):
        if self.wrap_arounds == 0:
            super().append(item)
        else:
            self.data[self.write_index] = item

        self.write_index = (self.write_index + 1) % self.capacity
        if self.write_index == 0:
            self.wrap_arounds += 1

    def sample(self):
        return random.randint(0, len(self.data) - 1)
