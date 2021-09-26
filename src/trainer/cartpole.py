import os
from queue import Empty
import psutil
from itertools import count
import argparse
import gym
from torch.utils.tensorboard import SummaryWriter
import datetime
import os


import torch
import torch.multiprocessing as mp

from .replay.prioritized import PrioritizedMemory
from .learner import Learner
from .actor import Actor as BaseActor, Rollout as BaseRollout
from .settings import BATCH_SIZE, MEMORY_SIZE, C, H, W, NUM_STEPS, NUM_ACTORS
from .process import transition_from_memory
from .utils import do_every


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


def memory_metrics(memory, step):
    yield ("scalar", "memory/length", len(memory), step.value)

    priorities = memory.get_all_priorities()
    yield ("histogram", "memory/priorities", priorities, step.value)

    num_adds = memory.get_total_added()
    yield ("scalar", "memory/num_adds", num_adds, step.value)

    num_updates = memory.get_total_updated()
    yield ("scalar", "memory/num_updates", num_updates, step.value)


def host_metrics(device, step):
    gpu_mem = torch.cuda.memory_allocated(device) / 1e6
    yield ("scalar", "host/gpu_memory_usage", gpu_mem, step.value)

    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / 1e6
    yield ("scalar", "host/cpu_memory_usage", cpu_mem, step.value)


def learner_metrics(learner, step):
    yield ("scalar", "learner/loss", learner.last_loss, step.value)

    state_dict = learner.online_dqn.state_dict()
    for tensor_name in state_dict:
        tag = f"learner/{tensor_name}"
        tensor = state_dict[tensor_name]
        yield ("histogram", tag, tensor, step.value)


def rollout_metrics(rollout, step):
    rewards = torch.tensor(rollout.reward_sequence)
    yield ("histogram", "rollout/episode_rewards", rewards, step.value)

    yield ("scalar", "rollout/episode_total_reward", rewards.sum(), step.value)

    episode_mean_value = rollout.mean_value
    yield ("scalar", "rollout/episode_mean_value", episode_mean_value, step.value)

    frames = torch.stack(rollout.frame_sequence, dim=1)
    fps = 4
    yield ("video", "rollout/episode", frames, step.value, fps)


def actor_metrics(actor, episode_idx, step):
    rewards = torch.tensor(actor.reward_sequence)
    yield ("histogram", "actor/episode_rewards", rewards, step.value)
    yield ("scalar", "actor/num_episodes", episode_idx, step.value)
    yield ("scalar", "actor/epsilon", actor.get_eps_threshold(step.value), step.value)


def run_actor(transition_queue, metric_queue, step):
    actor = Actor(step, device="cpu")
    for i_episode in count():
        for transition in actor.episode():
            transition_queue.put(transition)

        if i_episode % 20 == 19:
            for metric in actor_metrics(actor, i_episode, step):
                metric_queue.put(metric)


def run_writer(metric_queue: mp.Queue, name):
    timezone = datetime.timezone.utc
    current_date = datetime.datetime.now(timezone)
    version = current_date.strftime("d%Y_%m_%d-t%H_%M_%S")
    writer = SummaryWriter(os.path.join("../runs", name, version))

    while True:
        metric = metric_queue.get()
        m_type = metric[0]
        if m_type == "histogram":
            writer.add_histogram(*metric[1:])
        if m_type == "scalar":
            writer.add_scalar(*metric[1:])
        if m_type == "video":
            writer.add_video(*metric[1:])
        if m_type == "graph":
            writer.add_graph(*metric[1:])


def main():
    mp.set_start_method("spawn")

    step = mp.Value('i', 0)
    metric_queue = mp.Queue()
    metric_p = mp.Process(
        target=run_writer,
        args=(metric_queue, 'cartpole'))
    metric_p.start()

    device = get_preferred_device() if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print("Using:", device)

    for metric in host_metrics(device, step):
        metric_queue.put(metric)

    memory = PrioritizedMemory(MEMORY_SIZE, transform=transition_from_memory)
    learner = Learner(step, device=device)
    rollout = Rollout(step, model=learner.online_dqn, device=device)

    transition_queue = mp.Queue(maxsize=2 * BATCH_SIZE * NUM_ACTORS)
    actor_p = mp.Process(
        target=run_actor,
        args=(transition_queue, metric_queue, step))
    actor_p.start()

    state = rollout.env.reset()
    frame = rollout.get_frame(state)
    state = rollout.stacking(frame)
    metric_queue.put(("graph", learner.online_dqn, state))

    while True:
        if step.value >= NUM_STEPS:
            break

        if do_every(10_000, step.value):
            for metric in host_metrics(device, step):
                metric_queue.put(metric)
            for metric in memory_metrics(memory, step):
                metric_queue.put(metric)

        try:
            transition = transition_queue.get_nowait()
            memory.add(transition)
        except Empty as e:
            pass
        learner.step(memory, step)

        if do_every(500, step.value):
            for metric in learner_metrics(learner, step):
                metric_queue.put(metric)

        if do_every(2_500, step.value):
            for transition in rollout.episode():
                memory.add(transition)

            for metric in rollout_metrics(rollout, step):
                metric_queue.put(metric)

        if do_every(25_000, step.value):
            torch.save({
                'steps': step.value,
                'model_state_dict': learner.online_dqn.state_dict(),
                'optimizer_state_dict': learner.optimizer.state_dict()
            }, "../checkpoints/traininpt")

    rollout.env.close()
    actor_p.kill()
    actor_p.close()


if __name__ == "__main__":
    main()
