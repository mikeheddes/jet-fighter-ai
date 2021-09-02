import os
import time
import psutil
from itertools import count

import torch
import ray

from .replay.prioritized import PrioritizedMemory
from .learner import Learner
from .actor import Actor, Rollout
from .globals import Commons, MEMORY_SIZE, C, H, W, NUM_STEPS
from .process import transition_from_memory



def report_host_metrics(device):
    commons = ray.get_actor("commons")

    gpu_mem = torch.cuda.memory_allocated(device) / 1e6
    commons.write_scalar.remote("host/gpu_memory_usage", gpu_mem)

    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / 1e6
    commons.write_scalar.remote("host/cpu_memory_usage", cpu_mem)


def main():
    ray.init()

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    commons = Commons.options(name="commons").remote()
    memory = PrioritizedMemory.options(name="memory").remote(
        MEMORY_SIZE, transform=transition_from_memory)
    learner = Learner(device=device)
    actor = Actor(model=learner.online_dqn, device=device)
    rollout = Rollout(model=learner.online_dqn, device=device)

    report_host_metrics(device)

    state = actor.env.reset()
    state = torch.tensor(state, dtype=torch.float, device=device)
    state = state.view(1, C, H, W)
    state = actor.transform(state)
    state = actor.stacking(state)
    commons.write_graph.remote(learner.online_dqn, state)

    for i_episode in count():
        # TODO: Cache call and invalidate every n-calls
        step = ray.get(commons.get_step.remote())
        if step >= NUM_STEPS:
            break

        if i_episode % 20 == 19:
            report_host_metrics(device)
            actor.report_metrics(i_episode)
            memory.report_metrics.remote()

        for transition in actor.episode():
            memory.add.remote(transition)
            learner.step(memory)

        if i_episode % 5 == 4:
            for transition in rollout.episode():
                memory.add.remote(transition)

            rollout.report_metrics(i_episode)

        if i_episode % 50 == 49:
            step = ray.get(commons.get_step.remote())
            torch.save({
                'steps': step,
                'model_state_dict': learner.online_dqn.state_dict(),
                'optimizer_state_dict': learner.optimizer.state_dict()
            }, "../checkpoints/training.pt")

    actor.env.close()
    rollout.env.close()


if __name__ == "__main__":
    main()
