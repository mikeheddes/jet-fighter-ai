import random


class Memory:
    def __init__(self, capacity, alpha=0.6, beta=0.4, epsilon=0.001):
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.tree = SumTree(capacity)

        self.capacity = capacity
        self.wrap_arounds = 0
        self.write_index = 0
        self.data = []

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

    def __getitem__(self, index):
        return self.data[index]

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


class SumTree:
    """
    Basic implementation of a sum tree
    The parent of two nodes has the value of the sum of its children
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.nodes = [0.] * (capacity * 2 - 1)

    @property
    def sum(self):
        return self.nodes[0]

    def update(self, index, value):
        """
        Update the value of a leave node
        Args
            index: is the external index of the leave node
            value: the new value of the node
        """
        assert index < self.capacity, "Index out of range"

        internal_index = index + self.capacity - 1
        change = value - self.nodes[internal_index]

        self.nodes[internal_index] = value
        self.propagate(internal_index, change)

    def get(self, at_sum):
        """
        Gets the index and value of the node which reached the at_sum value
        Args
            at_sum: the summed value to be reached. Needs to be lower than the total sum of the tree.
        Returns
            index: the index of the leave node
            value: the value of the leave node
        """
        assert at_sum <= self.sum, "Value of at_sum cannot be larger than the sum of the tree"

        internal_index = self.retrieve(0, at_sum)
        index = internal_index - self.capacity + 1

        return index, self.nodes[internal_index]

    def propagate(self, index, change):
        parent = (index - 1) // 2
        self.nodes[parent] += change

        if parent != 0:
            self.propagate(parent, change)

    def retrieve(self, index, at_sum):
        left = 2 * index + 1
        right = left + 1

        if left >= len(self.nodes):
            return index

        if at_sum <= self.nodes[left]:
            return self.retrieve(left, at_sum)
        else:
            return self.retrieve(right, at_sum - self.nodes[left])
