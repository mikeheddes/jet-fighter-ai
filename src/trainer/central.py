import os
import psutil
from itertools import count

import torch

from .replay.prioritized import PrioritizedMemory
from .learner import Learner
from .actor import Actor, Rollout
from .globals import variables, writer, MEMORY_SIZE, C, H, W, NUM_STEPS
from .process import transition_from_memory


def report_memory_metrics(memory):
    step = variables.get_step()
    writer.add_scalar("memory/length", len(memory), step)

    priorities = memory.get_all_priorities()
    writer.add_histogram("memory/priorities", priorities, step)

    num_adds = memory.get_total_added()
    writer.add_scalar("memory/num_adds", num_adds, step)

    num_updates = memory.get_total_updated()
    writer.add_scalar("memory/num_updates", num_updates, step)


def report_host_metrics(device):
    step = variables.get_step()
    gpu_mem = torch.cuda.memory_allocated(device) / 1e6
    writer.add_scalar("host/gpu_memory_usage", gpu_mem, step)

    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / 1e6
    writer.add_scalar("host/cpu_memory_usage", cpu_mem, step)


def report_rollout_metrics(rollout):
    step = variables.get_step()
    rewards = torch.tensor(rollout.reward_sequence)
    writer.add_histogram("rollout/episode_rewards", rewards, step)

    writer.add_scalar("rollout/episode_total_reward", rewards.sum(), step)

    episode_mean_value = rollout.mean_value
    writer.add_scalar("rollout/episode_mean_value", episode_mean_value, step)

    frames = torch.stack(rollout.frame_sequence, dim=1)
    writer.add_video("rollout/episode", frames, step, fps=4)


def report_actor_metrics(actor, episode_idx):
    step = variables.get_step()
    rewards = torch.tensor(actor.reward_sequence)
    writer.add_histogram("actor/episode_rewards", rewards, step)

    writer.add_scalar("actor/num_episodes", episode_idx, step)
    writer.add_scalar("actor/epsilon", actor.get_eps_threshold(step), step)


def main():
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    report_host_metrics(device)

    memory = PrioritizedMemory(MEMORY_SIZE, transform=transition_from_memory)
    learner = Learner(device=device)
    actor = Actor(model=learner.online_dqn, device=device)
    rollout = Rollout(model=learner.online_dqn, device=device)

    state = actor.env.reset()
    state = torch.tensor(state, dtype=torch.float, device=device)
    state = state.view(1, C, H, W)
    state = actor.transform(state)
    state = actor.stacking(state)
    writer.add_graph(learner.online_dqn, state)

    for i_episode in count():
        if variables.get_step() >= NUM_STEPS:
            break

        if i_episode % 20 == 19:
            report_host_metrics(device)
            report_actor_metrics(actor, i_episode)

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
            }, "../checkpoints/training.pt")

    actor.env.close()
    rollout.env.close()
    writer.close()


if __name__ == "__main__":
    main()
