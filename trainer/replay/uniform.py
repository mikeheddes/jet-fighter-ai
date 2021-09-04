import random

class UniformMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.wrap_arounds = 0
        self.write_index = 0
        self.data = []

    def __len__(self):
        return len(self.data)

    def add(self, item):
        if self.wrap_arounds == 0:
            self.data.append(item)
        else:
            self.data[self.write_index] = item

        self.write_index = (self.write_index + 1) % self.capacity
        if self.write_index == 0:
            self.wrap_arounds += 1

    def sample(self, batch_size=1):
        return random.sample(self.data, batch_size)
