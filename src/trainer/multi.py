import base64
import io
import asyncio
import time
import random
from collections import namedtuple
from itertools import count
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torch.multiprocessing as multiprocessing

from trainer.env import Environment
from trainer.memory import Memory
from trainer.model import DQN


BATCH_SIZE = 28
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 200
LEARNING_RATE = 0.001
NUM_ACTORS = 1

# MESSAGE TYPES
SAMPLE_TRANSITIONS = 0
NEW_MODEL_WEIGHTS = 1

Transition = namedtuple(
    'Transition', ('state', 'action', 'reward', 'next_state'))


def start_learner_process(pipe_with_memory):
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    n_actions = 4

    dqn = DQN((1, 90, 120), n_actions).to(device)
    optimizer = optim.RMSprop(dqn.parameters(), lr=LEARNING_RATE)

    pipe_with_memory.send(SAMPLE_TRANSITIONS)
    while True:
        transitions = pipe_with_memory.recv()
        pipe_with_memory.send(SAMPLE_TRANSITIONS)
        batch = Transition(*zip(*transitions))

        non_final_mask = tuple(
            map(lambda s: s is not None, batch.next_state))
        non_final_mask = torch.tensor(
            non_final_mask, device=device, dtype=torch.bool)
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        q_values = dqn(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = dqn(
            non_final_next_states).amax(1).detach()
        expected_q_values = (next_state_values * GAMMA) + reward_batch

        loss = F.mse_loss(q_values, expected_q_values.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def worker(worker_idx, learner_pipe, rollout_pipe, actors_pipes):
    # Learner
    if worker_idx == 0:
        pipe_with_memory, _ = learner_pipe
        start_learner_process(pipe_with_memory)

    # Memory
    elif worker_idx == 1:
        _, pipe_with_learner = learner_pipe
        _, pipe_with_rollout = rollout_pipe
        pipes_with_actors = [actor_pipe[1] for actor_pipe in actors_pipes]
        start_memory_process(pipe_with_learner, pipe_with_rollout, pipes_with_actors)

    # Rollout
    elif worker_idx == 2:
        pipe_with_memory, _ = rollout_pipe
        start_rollout_process(pipe_with_memory)

    # Actors
    elif worker_idx > 2:
        pipe_with_memory, _ = actors_pipes[worker_idx - 3]
        start_actor_process(pipe_with_memory)



def main():

    learner_pipe = multiprocessing.Pipe()
    rollout_pipe = multiprocessing.Pipe()
    actors_pipes = [multiprocessing.Pipe() for _ in range(NUM_ACTORS)]

    multiprocessing.spawn(
        worker, 
        args=(learner_pipe, rollout_pipe, actors_pipes),
        nprocs=3 + NUM_ACTORS)




def main():
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    n_actions = 4

    dqn = DQN(state_size, n_actions).to(device)
    optimizer = optim.RMSprop(dqn.parameters(), lr=LEARNING_RATE)
    memory = Memory(10000)

    def select_action(state, step):
        sample = random.random()
        part = 1. - min(step / EPS_DECAY, 1.)
        eps_threshold = EPS_END + (EPS_START - EPS_END) * part
        if sample > eps_threshold:
            with torch.no_grad():
                return dqn(state).max(1)[1].view(1, 1)
        else:
            return torch.randint(0, n_actions, (1, 1), device=device, dtype=torch.long)

    env = Environment()
    step = 0
    for i_episode in range(5000):
        print(f"start episode {i_episode}")
        state = env.reset()
        state = state.permute(2, 0, 1).unsqueeze(0).float().to(device)
        for step in count(step):
            action = select_action(state, step)
            next_state, reward, done = env.step(action.item())
            reward = torch.tensor(
                [reward], dtype=torch.float, device=device)

            if done:
                next_state = None
            else:
                next_state = next_state.permute(
                    2, 0, 1).unsqueeze(0).float().to(device)

            transition = Transition(state, action, reward, next_state)
            memory.push(transition)

            state = next_state

            if len(memory) < 1000:
                return
            transitions = memory.sample(BATCH_SIZE)
            batch = Transition(*zip(*transitions))

            non_final_mask = tuple(
                map(lambda s: s is not None, batch.next_state))
            non_final_mask = torch.tensor(
                non_final_mask, device=device, dtype=torch.bool)
            non_final_next_states = torch.cat(
                [s for s in batch.next_state if s is not None])
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            q_values = dqn(state_batch).gather(1, action_batch)

            next_state_values = torch.zeros(BATCH_SIZE, device=device)
            next_state_values[non_final_mask] = dqn(
                non_final_next_states).amax(1).detach()
            expected_q_values = (next_state_values * GAMMA) + reward_batch

            loss = F.mse_loss(q_values, expected_q_values.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if done:
                if i_episode % 5 == 4:
                    state = env.reset()
                    state = state.permute(2, 0, 1).unsqueeze(0).float().to(device)
                    mean_max_q_value = 0
                    for t in count():
                        q_values = dqn(state)
                        action = q_values.max(1)[1].view(1, 1)
                        mean_max_q_value += q_values.max(1)[0].item()
                        next_state, reward, done = env.step(action.item())
                        reward = torch.tensor([reward], dtype=torch.float, device=device)

                        if done:
                            next_state = None
                        else:
                            next_state = next_state.permute(
                                2, 0, 1).unsqueeze(0).float().to(device)

                        transition = Transition(state, action, reward, next_state)
                        memory.push(transition)

                        state = next_state

                        if done:
                            break

                    episode_duration =  t + 1
                    mean_q_value = mean_max_q_value / (t + 1)
                    print("episode duration", episode_duration,
                            f"\tmean q-value: {mean_q_value:.1f}", "\tat episode", i_episode)

                if i_episode % 50 == 49:
                    torch.save({
                        'steps': step,
                        'model_state_dict': dqn.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    }, "checkpoints/model.pt")
                break


if __name__ == '__main__':
    main()
