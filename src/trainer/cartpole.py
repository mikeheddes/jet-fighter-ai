import os
import psutil
from itertools import count
import argparse
import gym

import torch

from .replay.prioritized import PrioritizedMemory
from .learner import Learner
from .actor import Actor as BaseActor, Rollout as BaseRollout
from .settings import variables, MEMORY_SIZE, C, H, W, NUM_STEPS
from .process import transition_from_memory

class CartPoleActor:
    def get_env(self):
        return gym.make('CartPole-v1')

    def get_frame(self, raw_state):
        state = torch.tensor(raw_state, dtype=torch.float, device=self.device)
        state = state.view(1, C, H, W)
        return state


class Actor(CartPoleActor, BaseActor):
    pass


class Rollout(CartPoleActor, BaseRollout):
    pass


def get_preferred_device():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", default="cuda:0")

    return parser.parse_args().device


def report_memory_metrics(memory):
    step = variables.get_step()
    variables.writer.add_scalar("memory/length", len(memory), step)

    priorities = memory.get_all_priorities()
    variables.writer.add_histogram("memory/priorities", priorities, step)

    num_adds = memory.get_total_added()
    variables.writer.add_scalar("memory/num_adds", num_adds, step)

    num_updates = memory.get_total_updated()
    variables.writer.add_scalar("memory/num_updates", num_updates, step)


def report_host_metrics(device):
    step = variables.get_step()
    gpu_mem = torch.cuda.memory_allocated(device) / 1e6
    variables.writer.add_scalar("host/gpu_memory_usage", gpu_mem, step)

    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / 1e6
    variables.writer.add_scalar("host/cpu_memory_usage", cpu_mem, step)


def report_rollout_metrics(rollout):
    step = variables.get_step()
    rewards = torch.tensor(rollout.reward_sequence)
    variables.writer.add_histogram("rollout/episode_rewards", rewards, step)

    variables.writer.add_scalar("rollout/episode_total_reward", rewards.sum(), step)

    episode_mean_value = rollout.mean_value
    variables.writer.add_scalar("rollout/episode_mean_value", episode_mean_value, step)

    frames = torch.stack(rollout.frame_sequence, dim=1)
    variables.writer.add_video("rollout/episode", frames, step, fps=4)


def report_actor_metrics(actor, episode_idx):
    step = variables.get_step()
    rewards = torch.tensor(actor.reward_sequence)
    variables.writer.add_histogram("actor/episode_rewards", rewards, step)

    variables.writer.add_scalar("actor/num_episodes", episode_idx, step)
    variables.writer.add_scalar("actor/epsilon", actor.get_eps_threshold(step), step)


def main():
    variables.init_writer('cartpole')

    device = get_preferred_device() if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print("Using:", device)

    report_host_metrics(device)

    memory = PrioritizedMemory(MEMORY_SIZE, transform=transition_from_memory)
    learner = Learner(device=device)
    actor = Actor(model=learner.online_dqn, device=device)
    rollout = Rollout(model=learner.online_dqn, device=device)

    state = actor.env.reset()
    frame = actor.get_frame(state)
    state = actor.stacking(frame)
    variables.writer.add_graph(learner.online_dqn, state)

    for i_episode in count():
        if variables.get_step() >= NUM_STEPS:
            break

        if i_episode % 20 == 19:
            report_host_metrics(device)
            report_actor_metrics(actor, i_episode)
            report_memory_metrics(memory)

        for transition in actor.episode():
            memory.add(transition)
            learner.step(memory)

        if i_episode % 5 == 4:
            for transition in rollout.episode():
                memory.add(transition)

            report_rollout_metrics(rollout)

        if i_episode % 50 == 49:
            torch.save({
                'steps': variables.get_step(),
                'model_state_dict': learner.online_dqn.state_dict(),
                'optimizer_state_dict': learner.optimizer.state_dict()
            }, "../checkpoints/traininpt")

    actor.env.close()
    rollout.env.close()
    variables.writer.close()


if __name__ == "__main__":
    main()
