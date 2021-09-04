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
        self.action_sequence = []
        self.frame_idx = 0

        self.commons = ray.get_actor("commons")

        self.get_step_countdown = 0

    def get_step(self):
        if self.get_step_countdown <= 0:
            self.get_step_countdown = 100
            self.cached_step = ray.get(self.commons.get_step.remote())

        self.get_step_countdown -= 1
        return self.cached_step

    def get_eps_threshold(self, step):
        part = 1. - min(step / EPS_DECAY, 1.)
        return EPS_END + (EPS_START - EPS_END) * part

    def policy(self, state):
        sample = random.random()
        step = self.get_step()
        if sample > self.get_eps_threshold(step):
            with torch.no_grad():
                return self.model(state).argmax().item()
        else:
            return random.randint(0, NUM_ACTIONS - 1)

    def episode(self):
        self.reward_sequence = []
        self.frame_sequence = []
        self.action_sequence = []
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
            self.action_sequence.append(action)

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
        actions = torch.tensor(self.action_sequence)
        self.commons.write_histogram.remote("actor/episode_actions", actions)
        self.commons.write_scalar.remote("actor/num_episodes", episode_idx)

        step = self.get_step()
        self.commons.write_scalar.remote(
            "actor/epsilon", self.get_eps_threshold(step))

    def run(self):
        memory = ray.get_actor("memory")

        for i_episode in count():
            step = self.get_step()
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
        rewards = torch.tensor(self.reward_sequence)
        self.commons.write_histogram.remote("rollout/episode_rewards", rewards)
        actions = torch.tensor(self.action_sequence)
        self.commons.write_histogram.remote("rollout/episode_actions", actions)

        self.commons.write_scalar.remote(
            "rollout/episode_total_reward", rewards.sum())
        self.commons.write_scalar.remote(
            "rollout/episode_mean_value", self.mean_value)

        frames = torch.stack(self.frame_sequence, dim=1)
        self.commons.write_video.remote("rollout/episode", frames, fps=4)
