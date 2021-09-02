import torch
import random
import ray

from .sumtree import SumTree


@ray.remote
class PrioritizedMemory:
    def __init__(self, capacity, alpha=0.6, beta=0.4, epsilon=0.001, transform=None):
        self.transform = transform
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.tree = SumTree(capacity)

        self.capacity = capacity
        self.wrap_arounds = 0
        self.write_index = 0
        self.data = []

        # metrics
        self.num_updates = 0
        self.commons = ray.get_actor("commons")

    def __len__(self):
        return len(self.data)

    def add(self, item, error=None):
        if self.wrap_arounds == 0:
            self.data.append(item)
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

    def sample(self, batch_size=1):
        transitions = [None] * batch_size
        sample_ids = [None] * batch_size
        is_weights = torch.empty(batch_size, dtype=torch.float)

        for i in range(batch_size):
            at_sum = random.random() * self.tree.sum

            index, priority = self.tree.get(at_sum)

            if self.transform:
                transitions[i] = self.transform(self, index)
            else:
                transitions[i] = self.data[index]

            sample_ids[i] = index + self.wrap_arounds * self.capacity
            probability = priority / self.tree.sum
            is_weights[i] = (len(self.data) * probability) ** -self.beta

        is_weights /= is_weights.max()
        is_weights = is_weights.unsqueeze(1)

        return transitions, sample_ids, is_weights

    def get_priority(self, error):
        return (abs(error) + self.epsilon) ** self.alpha

    def update_priority(self, sample_id, error):
        self.num_updates += 1

        items_added = self.wrap_arounds * self.capacity + self.write_index - 1

        # Stop if the item to update is no longer in memory
        if sample_id < items_added - self.capacity:
            return

        index = self.sample_id_to_index(sample_id)

        priority = self.get_priority(error)
        self.tree.update(index, priority)

    def update_priorities(self, sample_ids, errors):
        for sample_id, error in zip(sample_ids, errors):
            self.update_priority(sample_id, error)

    def sample_id_to_index(self, sample_id):
        return sample_id % self.capacity

    def get_all_priorities(self):
        priorities = self.tree.nodes[-self.tree.capacity:]
        return torch.tensor(priorities, dtype=torch.float)

    def get_total_added(self):
        return self.wrap_arounds * self.capacity + self.write_index - 1

    def get_total_updated(self):
        return self.num_updates

    def report_metrics(self):
        self.commons.write_scalar.remote("memory/length", len(self))

        priorities = self.get_all_priorities()
        self.commons.write_histogram.remote("memory/priorities", priorities)

        num_adds = self.get_total_added()
        self.commons.write_scalar.remote("memory/num_adds", num_adds)

        num_updates = self.get_total_updated()
        self.commons.write_scalar.remote("memory/num_updates", num_updates)
