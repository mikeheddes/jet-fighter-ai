from collections import deque, namedtuple
import torch
import torchvision.transforms as T

from trainer.types import Transition


class Transform(torch.nn.Module):
    def __init__(self, frame_size):
        super().__init__()
        self.transform = torch.nn.Sequential(
            # T.Grayscale(),
            T.Resize(frame_size))

    def forward(self, x):
        return self.transform(x)


class Stacking:
    def __init__(self, num_frames):
        super().__init__()
        self.buffer = deque(maxlen=num_frames)
        self.num_frames = num_frames

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

    def from_memory(self, memory, batch_size=1):
        transitions = []
        for _ in range(batch_size):
            idx = memory.sample()
            min_idx = max(idx - self.num_frames + 1, 0)
            transition_stack = memory[min_idx:idx + 1]

            if len(transition_stack) > 1:
                for i in range(len(transition_stack) - 2, -1, -1):
                    if transition_stack[i].next_state == None:
                        transition_stack = transition_stack[i+1:]
                        break

            state = [t.state for t in transition_stack]
            if transition_stack[-1].next_state == None:
                next_state = None
            else:
                next_state = [t.next_state for t in transition_stack]

            transitions.append(
                Transition(
                    self.stack(state),
                    transition_stack[-1].action,
                    transition_stack[-1].reward,
                    self.stack(next_state)
                )
            )

        return transitions


def get_prediction_and_target(batch, online_dqn, target_dqn, batch_size=1, gamma=0.99, device=None):
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
        expected_q_values = reward_batch + (next_state_values * gamma)

    return q_values, expected_q_values
