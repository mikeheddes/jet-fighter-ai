from .sum_tree import SumTree

import random
from collections import UserList


class PrioritizedReplay(UserList):
    def __init__(self, capacity, alpha=0.6, beta=0.4, epsilon=0.001):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.tree = SumTree(capacity)

        self.capacity = capacity
        self.wrap_arounds = 0
        self.write_index = 0

    def append(self, item, error=None):
        if self.wrap_arounds == 0:
            super().append(item)
        else:
            self.data[self.write_index] = item

        # Set average error if None
        if error is None:
            error = (self.tree.sum / len(self.data)) ** (1 / self.alpha)

        priority = self.get_priority(error)
        self.tree.update(self.write_index, priority)

        # Update the next write index
        self.write_index = (self.write_index + 1) % self.capacity
        if self.write_index == 0:
            self.wrap_arounds += 1

    def sample(self):
        at_sum = random.random() * self.tree.sum

        index, priority = self.tree.get(at_sum)
        sample_id = index + self.wrap_arounds * self.capacity
        probability = priority / self.tree.sum
        is_weight = (len(self.data) * probability) ** -self.beta

        return sample_id, is_weight

    def get_priority(self, error):
        return (abs(error) + self.epsilon) ** self.alpha

    def update_priority(self, sample_id, error):
        items_added = self.wrap_arounds * self.capacity + self.write_index - 1

        # Stop if the item to update is no longer in memory
        if sample_id < items_added - self.capacity:
            return

        index = self.sample_id_to_index(sample_id)

        priority = self.get_priority(error)
        self.tree.update(index, priority)

    def sample_id_to_index(self, sample_id):
        return sample_id % self.capacity
