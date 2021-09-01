import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

from .types import Transition
from .globals import STACKING


def stack(frames, minlen=1):
    padding = []
    for _ in range(max(0, minlen - len(frames))):
        padding.append(torch.zeros(
            size=frames[0].shape,
            dtype=frames[0].dtype,
            device=frames[0].device))

    frames = padding + frames
    return torch.cat(frames, dim=1)


def get_state_from_transitions(transitions):
    start_idx = 0
    for i in range(len(transitions) - 2, -1, -1):
        if transitions.next_state is None:
            start_idx = i + 1

    state = [t.state for t in transitions[start_idx:]]
    return stack(state, STACKING)


def get_next_state_from_transitions(transitions):
    if transitions[-1].next_state is None:
        return None

    start_idx = 0
    for i in range(len(transitions) - 2, -1, -1):
        if transitions.next_state is None:
            start_idx = i + 1

    next_state = [t.next_state for t in transitions[start_idx:]]
    return stack(next_state, STACKING)


def transition_from_memory(memory, index):
    start_idx = max(0, index - STACKING + 1)
    end_idx = index + 1

    transition_stack = memory.data[start_idx:end_idx]
    return Transition(
        state=get_state_from_transitions(transition_stack),
        action=transition_stack[-1].action,
        reward=transition_stack[-1].reward,
        next_state=get_next_state_from_transitions(transition_stack))


class Grayscale(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.tensor(
            [[[[1.0]]]],
            dtype=torch.float), requires_grad=False)

    def forward(self, x):
        return (x * self.weights).sum(1, keepdims=True)


class Downscale(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weights = nn.Parameter(torch.full(
            (channels, channels, 1, 1), 1.0,
            dtype=torch.float), requires_grad=False)

    def forward(self, x):
        return F.conv2d(x, weight=self.weights, stride=1)


class Transform(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gray = Grayscale()
        self.downscaling = Downscale(channels)

    def forward(self, x):
        out = self.gray(x)
        out = self.downscaling(out)
        return out


class StackingBuffer:
    def __init__(self, num_frames):
        super().__init__()
        self.num_frames = num_frames
        self.buffer = deque(maxlen=num_frames)

    def reset(self):
        self.buffer.clear()

    def __call__(self, frame):
        self.buffer.append(frame)
        return stack(list(self.buffer), self.num_frames)
