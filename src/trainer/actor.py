import gym
import ray
import random
from itertools import count
import torch

from .model import DQN
from .types import Transition
from .globals import STACKING, C, H, W, NUM_ACTIONS, EPS_DECAY, EPS_START, EPS_END, NUM_STEPS
from .process import Transform, StackingBuffer


class Actor:
    def __init__(self, model=None, device=None):
        self.device = device
        self.env = gym.make('CartPole-v1')

        if model is None:
            self.model = DQN(
                (STACKING * C, H, W),
                NUM_ACTIONS).to(device)
        else:
            self.model = model

        self.transform = Transform(C).to(device)
        self.stacking = StackingBuffer(STACKING)
        self.reward_sequence = []
        self.frame_sequence = []
        self.frame_idx = 0

        self.commons = ray.get_actor("commons")

    def get_eps_threshold(self, step):
        part = 1. - min(step / EPS_DECAY, 1.)
        return EPS_END + (EPS_START - EPS_END) * part

    def policy(self, state):
        sample = random.random()

        # TODO: Cache call and invalidate every n-calls
        step = ray.get(self.commons.get_step.remote())
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
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        state = state.view(1, C, H, W)
        state = self.transform(state)
        frame = state
        state = self.stacking(state)

        while True:
            action = self.policy(state)
            next_state, reward, done, _ = self.env.step(action)

            self.reward_sequence.append(reward)
            self.frame_sequence.append(frame)

            if done:
                next_state = None
            else:
                next_state = torch.tensor(
                    next_state, dtype=torch.float, device=self.device)
                next_state = next_state.view(1, C, H, W)
                next_state = self.transform(next_state)
                next_frame = next_state
                next_state = self.stacking(next_state)

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

    def report_metrics(self, episode_idx):
        rewards = torch.tensor(self.reward_sequence)
        self.commons.write_histogram.remote("actor/episode_rewards", rewards)

        self.commons.write_scalar.remote("actor/num_episodes", episode_idx)

        step = ray.get(self.commons.get_step.remote())
        self.commons.write_scalar.remote(
            "actor/epsilon", self.get_eps_threshold(step))

    def run(self):
        memory = ray.get_actor("memory")

        for i_episode in count():
            # TODO: Cache call and invalidate every n-calls
            step = ray.get(self.commons.get_step.remote())
            if step >= NUM_STEPS:
                break

            if i_episode % 20 == 19:
                self.report_metrics(i_episode)

            for transition in self.episode():
                # TODO: Buffer transitions locally and send in batches
                memory.add.remote(transition)


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

    def report_metrics(self, episode_idx):
        commons = ray.get_actor("commons")

        rewards = torch.tensor(self.reward_sequence)
        commons.write_histogram.remote("rollout/episode_rewards", rewards)

        commons.write_scalar.remote(
            "rollout/episode_total_reward", rewards.sum())
        commons.write_scalar.remote(
            "rollout/episode_mean_value", self.mean_value)

        frames = torch.stack(self.frame_sequence, dim=1)
        commons.write_video.remote("rollout/episode", frames, fps=4)
