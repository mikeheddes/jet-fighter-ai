class Transition:
    __slots__ = ['state', 'action', 'reward', 'next_state']

    def __init__(self, state, action, reward, next_state):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state

    def __iter__(self):
        as_tuple = (self.state, self.action, self.reward, self.next_state)
        return iter(as_tuple)


class TransitionBatch:
    __slots__ = ['state', 'action', 'reward', 'next_state']

    def __init__(self, state, action, reward, next_state):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
