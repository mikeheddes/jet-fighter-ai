from queue import Empty
import gym

import torch
import torch.multiprocessing as mp

from .replay.prioritized import PrioritizedMemory
from .learner import Learner
from .actor import Actor as BaseActor, Rollout as BaseRollout, run_actor
from .settings import BATCH_SIZE, MEMORY_SIZE, C, H, MIN_MEMORY_SIZE, W, NUM_STEPS, NUM_ACTORS, STACKING, NUM_ACTIONS
from .process import transition_from_memory
from .utils import do_every, run_writer, get_preferred_device, host_metrics
from .model import LinearDQN as DQN
from .types import Transition


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


def main():
    mp.set_start_method("spawn")

    step = mp.Value('i', 0)
    metric_q = mp.Queue()
    metric_p = mp.Process(
        target=run_writer,
        args=(metric_q, 'cartpole'))
    metric_p.start()

    device = get_preferred_device() if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print("Using:", device)

    for metric in host_metrics(device, step):
        metric_q.put(metric)

    memory = PrioritizedMemory(MEMORY_SIZE, transform=transition_from_memory)
    learner = Learner(DQN, step, device=device)
    rollout = Rollout(DQN, step, model=learner.online_dqn, device=device)

    transition_q = mp.Queue(maxsize=2 * BATCH_SIZE * NUM_ACTORS)
    actor_p = mp.Process(
        target=run_actor,
        args=(Actor, DQN, transition_q, metric_q, step))
    actor_p.start()

    state = rollout.env.reset()
    frame = rollout.get_frame(state)
    state = rollout.stacking(frame)
    metric_q.put(("graph", DQN((1, C * STACKING, H, W), NUM_ACTIONS), state))

    while True:
        if step.value >= NUM_STEPS:
            break

        if do_every(10_000, step.value):
            for metric in host_metrics(device, step):
                metric_q.put(metric)
            for metric in memory.metrics(step):
                metric_q.put(metric)

        try:
            t = transition_q.get_nowait()
            # Copy tensors to local memory
            memory.add(Transition(
                state=t.state.clone(),
                action=t.action,
                reward=t.reward,
                next_state=t.next_state.clone() if t.next_state != None else None
            ))
        except Empty:
            pass

        if len(memory) > MIN_MEMORY_SIZE:
            transitions, sample_ids, is_weights = memory.sample(BATCH_SIZE)
            batch = Transition(*zip(*transitions))
            errors = learner.step(batch, is_weights)
            for batch_i in range(BATCH_SIZE):
                memory.update_priority(
                    sample_ids[batch_i], errors[batch_i].item())

        if do_every(500, step.value):
            for metric in learner.metrics(step):
                metric_q.put(metric)

        if do_every(1_000, step.value):
            for transition in rollout.episode():
                memory.add(transition)

            for metric in rollout.metrics(step):
                metric_q.put(metric)

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
