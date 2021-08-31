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
