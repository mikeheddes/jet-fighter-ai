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
    os.makedirs("./checkpoints/", exist_ok=True)

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    commons = Commons.options(name="commons").remote()
    memory = PrioritizedMemory.options(name="memory").remote(
        MEMORY_SIZE, transform=transition_from_memory)
    learner = Learner.options(name="learner").remote(device=device)
    actor = Actor(device="cpu")
    rollout = Rollout(device="cpu")

    report_host_metrics(device)

    state = actor.env.reset()
    state = torch.tensor(state, dtype=torch.float, device=device)
    state = state.view(1, C, H, W)
    state = actor.transform(state)
    state = actor.stacking(state)
    commons.write_graph.remote(actor.model, state)

    learner.train.remote()

    for i_episode in count():
        # TODO: Cache call and invalidate every n-calls
        # step = ray.get(commons.get_step.remote())
        # if step >= NUM_STEPS:
        #     break

        if i_episode % 20 == 19:
            report_host_metrics(device)
            actor.report_metrics(i_episode)
            memory.report_metrics.remote()

        transitions = list(actor.episode())
        memory.add_batch.remote(transitions)
        time.sleep(0.1)

        if i_episode % 5 == 4:
            model_state_dict = ray.get(commons.get_model_state_dict.remote())
            rollout.update_model(model_state_dict)
            actor.update_model(model_state_dict)

            transitions = list(rollout.episode())
            memory.add_batch.remote(transitions)
            rollout.report_metrics(i_episode)
            time.sleep(0.1)

    actor.env.close()
    rollout.env.close()


if __name__ == "__main__":
    main()
