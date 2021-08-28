import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

from trainer.types import Transition


class Grayscale(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.tensor(
            [[[[0.2989]], [[0.587]], [[0.114]]]],
            dtype=torch.float)

    def forward(self, x):
        return (x * self.weights).sum(1, keepdims=True)


class Downscale(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weights = torch.full(
            (channels, channels, 2, 2),
            0.25,
            dtype=torch.float)

    def forward(self, x):
        return F.conv2d(x, weight=self.weights, stride=2)


class Transform(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gray = Grayscale()
        self.downscaling = Downscale(channels)

    def forward(self, x):
        out = self.gray(x)
        out = self.downscaling(out)
        return out


class Stacking:
    def __init__(self, num_frames, multi_step=1, gamma=0.99):
        super().__init__()
        self.buffer = deque(maxlen=num_frames)
        self.num_frames = num_frames
        self.multi_step = multi_step
        steps = torch.arange(0, multi_step)
        self.discounts = torch.pow(gamma, steps).unsqueeze(1)

    def reset(self):
        self.buffer.clear()

    def __call__(self, frame):
        self.buffer.append(frame)
        frames = list(self.buffer)
        return self.stack(frames)

    def stack(self, frames):
        assert len(frames) <= self.num_frames, "More than max frames provided"
        assert len(frames) >= 1, "Must provide at least one frame"

        if len(frames) < self.num_frames:
            padding = []
            num_padding_frames = self.buffer.maxlen - len(frames)
            for _ in range(num_padding_frames):
                empty_frame = torch.zeros(
                    size=frames[0].shape,
                    dtype=frames[0].dtype,
                    device=frames[0].device)
                padding.append(empty_frame)

            frames = padding + frames

        return torch.cat(frames, dim=1)

    def remove_episode_bleed_in(self, transition_stack):
        if len(transition_stack) > 1:
            for i in range(len(transition_stack) - 2, -1, -1):
                if transition_stack[i].next_state == None:
                    transition_stack = transition_stack[i + 1:]
                    break

        return transition_stack

    def truncate_multi_step(self, transition_stack):
        for i in range(len(transition_stack)):
            if transition_stack[i].next_state == None:
                transition_stack = transition_stack[:i + 1]
                break

        return transition_stack

    def extract_rewards(self, transition_stack):
        rewards = [t.reward for t in transition_stack]

        padding_len = self.multi_step - len(rewards)
        if padding_len != 0:
            zero = torch.zeros((1, 1), dtype=torch.float)
            rewards += [zero] * padding_len

        return torch.cat(rewards)

    def from_memory(self, memory, batch_size=1):
        transitions = []
        sample_ids = []
        is_weights = []

        for _ in range(batch_size):
            sample_id, is_weight = memory.sample()

            sample_ids.append(sample_id)
            is_weights.append(is_weight)

            # Create the stack of frames for the start state,
            # the state at which the action was taken.
            idx = memory.sample_id_to_index(sample_id)
            start_idx = max(idx - self.num_frames + 1, 0)
            start_stack = memory[start_idx:idx + 1]
            # Remove steps that are from a previous episode
            start_stack = self.remove_episode_bleed_in(start_stack)

            # Capture the N next steps
            end_idx = idx + self.multi_step
            multi_step_stack = memory[idx:end_idx]
            # Remove steps that are from the next episode
            multi_step_stack = self.truncate_multi_step(multi_step_stack)
            rewards = self.extract_rewards(multi_step_stack)
            rewards = rewards * self.discounts

            # Create the stack of frames for the next state,
            # the state the agent is in after the N-steps.
            num_frames_to_add = self.num_frames - len(multi_step_stack)
            frame_slice = slice(-num_frames_to_add - 1, -1)
            end_stack = start_stack[frame_slice] + multi_step_stack

            state = [t.state for t in start_stack]
            if end_stack[-1].next_state == None:
                next_state = None
            else:
                next_state = [t.next_state for t in end_stack]

            transitions.append(
                Transition(
                    self.stack(state),
                    start_stack[-1].action,
                    rewards.sum(0, keepdim=True),
                    self.stack(next_state)))

        is_weights = torch.tensor(is_weights, dtype=torch.float).unsqueeze(1)
        is_weights /= is_weights.max()

        return transitions, sample_ids, is_weights


def get_prediction_and_target(batch, online_dqn, target_dqn, batch_size=1, multi_step=1, gamma=0.99, device=None):
    non_final_mask = [s is not None for s in batch.next_state]
    non_final_mask = torch.tensor(
        non_final_mask, dtype=torch.bool, device=device)
    non_final_next_states = torch.cat(
        [s for s in batch.next_state if s is not None]).to(device)

    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    # Current prediction
    q_values = online_dqn(state_batch).gather(1, action_batch)

    # Calculate target
    with torch.no_grad():
        non_final_next_actions = online_dqn(
            non_final_next_states).argmax(1, keepdims=True)
        next_state_values = torch.zeros((batch_size, 1), device=device)
        next_state_values[non_final_mask] = target_dqn(
            non_final_next_states).gather(1, non_final_next_actions)
        discount = gamma ** multi_step
        expected_q_values = reward_batch + (next_state_values * discount)

    return q_values, expected_q_values
