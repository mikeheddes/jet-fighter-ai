import gym
import random
import torch

from .model import DQN
from .types import Transition
from .settings import variables, STACKING, C, H, W, NUM_ACTIONS, EPS_DECAY, EPS_START, EPS_END
from .process import StackingBuffer


class Actor:
    def __init__(self, model=None, device=None):
        self.device = device
        self.env = self.get_env()

        if model is None:
            self.model = DQN(
                (STACKING * C, H, W),
                NUM_ACTIONS).to(device)
        else:
            self.model = model

        self.stacking = StackingBuffer(STACKING)
        self.reward_sequence = []
        self.frame_sequence = []
        self.frame_idx = 0

    def get_env(self):
        raise NotImplementedError

    def get_frame(self, raw_state):
        raise NotImplementedError

    def get_eps_threshold(self, step):
        part = 1. - min(step / EPS_DECAY, 1.)
        return EPS_END + (EPS_START - EPS_END) * part

    def policy(self, state):
        sample = random.random()
        step = variables.get_step()
        if sample > self.get_eps_threshold(step):
            with torch.no_grad():
                return self.model(state).argmax().item()
        else:
            return random.randint(0, NUM_ACTIONS - 1)

    def episode(self):
        self.reward_sequence = []
        self.frame_sequence = []
        self.frame_idx = 0

        self.stacking.reset()
        state = self.env.reset()
        frame = self.get_frame(state)
        state = self.stacking(frame)

        while True:
            action = self.policy(state)
            next_state, reward, done, _ = self.env.step(action)

            self.reward_sequence.append(reward)
            self.frame_sequence.append(frame)

            if done:
                next_state = None
                next_frame = None
            else:
                next_frame = self.get_frame(next_state)
                next_state = self.stacking(next_frame)

            yield Transition(
                frame.cpu(),
                action,
                reward,
                next_frame.cpu() if next_state is not None else next_state)

            if done:
                return

            self.frame_idx += 1
            state = next_state
            frame = next_frame

    def update_model(self, state_dict):
        self.model.load_state_dict(state_dict)


class Rollout(Actor):
    def policy(self, state):
        with torch.no_grad():
            q_values = self.model(state)
            self.total_value += q_values.mean().item()
            return q_values.argmax().item()

    def episode(self):
        self.total_value = 0
        yield from super().episode()

    @property
    def mean_value(self):
        return self.total_value / self.frame_idx
