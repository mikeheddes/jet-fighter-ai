import random
import torch
from itertools import count

from .types import Transition
from .settings import STACKING, C, H, W, NUM_ACTIONS, EPS_DECAY
from .process import StackingBuffer

EPS_START = 1.0
EPS_END = 0.05


class Actor:
    def __init__(self, dqn_cls, step, model=None, device=None):
        self.device = device
        self.step = step

        if model is None:
            self.model = dqn_cls(
                (STACKING * C, H, W),
                NUM_ACTIONS).to(device)
        else:
            self.model = model

        self.stacking = StackingBuffer(STACKING)
        self.reward_sequence = []
        self.frame_sequence = []
        self.frame_idx = 0

        self.env = self.get_env()

    def get_env(self):
        raise NotImplementedError

    def get_frame(self, raw_state):
        raise NotImplementedError

    def get_eps_threshold(self, step):
        part = 1. - min(step / EPS_DECAY, 1.)
        return EPS_END + (EPS_START - EPS_END) * part

    def policy(self, state):
        sample = random.random()
        if sample > self.get_eps_threshold(self.step.value):
            with torch.no_grad():
                state = state.to(self.device)
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

    def metrics(self, episode_idx, step):
        rewards = torch.tensor(self.reward_sequence)
        yield ("histogram", "actor/episode_rewards", rewards, step.value)
        yield ("scalar", "actor/num_episodes", episode_idx, step.value)
        yield ("scalar", "actor/epsilon", self.get_eps_threshold(step.value), step.value)


def run_actor(actor_cls, dqn_cls, transition_queue, metric_queue, step):
    actor = actor_cls(dqn_cls, step, device="cpu")
    
    for i_episode in count():
        for transition in actor.episode():
            transition_queue.put(transition)

        if i_episode % 20 == 19:
            for metric in actor.metrics(i_episode, step):
                metric_queue.put(metric)


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

    def metrics(self, step, fps=4):
        rewards = torch.tensor(self.reward_sequence)
        yield ("histogram", "rollout/episode_rewards", rewards, step.value)

        yield ("scalar", "rollout/episode_total_reward", rewards.sum(), step.value)

        episode_mean_value = self.mean_value
        yield ("scalar", "rollout/episode_mean_value", episode_mean_value, step.value)

        frames = torch.stack(self.frame_sequence, dim=1)
        yield ("video", "rollout/episode", frames, step.value, fps)

